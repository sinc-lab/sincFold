from torch import nn
from torch.nn.functional import cross_entropy
import torch as tr
from tqdm import tqdm
import pandas as pd
import math

from sincfold.metrics import contact_f1
from sincfold.utils import mat2bp, postprocessing

SINCFOLD_WEIGHTS = 'https://github.com/sinc-lab/sincFold/raw/main/weights/weights.pmt'

def sincfold(pretrained=False, weights=None, **kwargs):
    """ 
    SincFold: a deep learning-based model for RNA secondary structure prediction
    pretrained (bool): Use pretrained weights
    **kwargs: Model hyperparameters
    """
    model = SincFold(**kwargs)
    if pretrained:
        print("Load pretrained weights...")
        model.load_state_dict(tr.hub.load_state_dict_from_url(SINCFOLD_WEIGHTS, map_location=tr.device(model.device)))
    else:
        if weights is not None:
            print(f"Load weights from {weights}")
            model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
        else:
            print("No weights provided, using random initialization")
        
    return model

class SincFold(nn.Module):
    def __init__(
        self,
        embedding_dim=4,
        device="cpu",
        negative_weight=0.1,
        lr=0.001,
        loss_l1=0,
        loss_beta=0,
        use_scheduler=False,
        verbose=True,
        **kwargs
    ):
        """Base classifier model from embedding sequence-
        negative_weigth: not_conected/conected proportion in error weights."""
        super().__init__()

        self.device = device
        self.class_weight = tr.tensor([negative_weight, 1.0]).float().to(device)
        self.loss_l1 = loss_l1
        self.loss_beta = loss_beta
        self.verbose = verbose
        self.config = kwargs

        mid_ch = 1
        if self.config["use_restrictions"]:
            mid_ch = 2

        # Define architecture
        self.build_graph(embedding_dim, mid_ch=mid_ch, **kwargs)

        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        # lr scheduler
        if use_scheduler:
            self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, verbose=True
            )
        else:
            self.scheduler = None

        self.to(device)
  
    def build_graph(
        self,
        embedding_dim,
        kernel=3,
        filters=16,
        num_layers=1,
        dilation_resnet1d=3,
        resnet_bottleneck_factor=0.5,
        mid_ch=1,
        kernel_resnet2d=5,
        bottleneck1_resnet2d=128,
        bottleneck2_resnet2d=64,
        filters_resnet2d=128,
        rank=32,
        dilation_resnet2d=3,
        **kwargs
    ):
        pad = (kernel - 1) // 2

        self.use_restrictions = mid_ch != 1

        self.resnet = [nn.Conv1d(embedding_dim, filters, kernel, padding="same")]

        for k in range(num_layers):
            self.resnet.append(
                ResidualLayer1D(
                    dilation_resnet1d,
                    resnet_bottleneck_factor,
                    filters,
                    kernel,
                )
            )

        self.resnet = nn.Sequential(*self.resnet)

        self.convsal1 = nn.Conv1d(
            in_channels=filters,
            out_channels=rank,
            kernel_size=kernel,
            padding=pad,
            stride=1,
        )
        self.convsal2 = nn.Conv1d(
            in_channels=filters,
            out_channels=rank,
            kernel_size=kernel,
            padding=pad,
            stride=1,
        )

        self.conv2D1 = nn.Conv2d(
            in_channels=mid_ch, out_channels=filters_resnet2d, kernel_size=7, padding="same"
        )
        self.resnet_block = [
            ResidualBlock2D(
                filters_resnet2d,
                bottleneck1_resnet2d,
                kernel_resnet2d,
                dilation_resnet2d,
            ), ResidualBlock2D(
                filters_resnet2d,
                bottleneck2_resnet2d,
                kernel_resnet2d,
                dilation_resnet2d,
            )
        ]
        
        self.resnet_block = nn.Sequential(*self.resnet_block)

        self.conv2Dout = nn.Conv2d(
            in_channels=filters_resnet2d,
            out_channels=1,
            kernel_size=kernel_resnet2d,
            padding="same",
        )

    def forward(self, x, *args):
        """args includes additional variables from dataloader: L, mask, seqid, sequence, [prob_mask]"""
        n = x.shape[2]
        mask = args[1].to(self.device)
        y = self.resnet(x)
        ya = self.convsal1(y)
        ya = tr.transpose(ya, -1, -2)

        yb = self.convsal2(y)

        y = ya @ yb
        yt = tr.transpose(y, -1, -2)
        y = (y + yt) / 2

        y0 = y.view(-1, n, n).multiply(mask)  # add valid connection mask

        batch_size = x.shape[0]

        if self.use_restrictions:
            prob_mat = args[4].to(self.device)
            x1 = tr.zeros([batch_size, 2, n, n]).to(self.device)
            x1[:, 0, :, :] = y0
            x1[:, 1, :, :] = prob_mat
        else:
            x1 = y0.unsqueeze(1)

        # Representation
        y = self.conv2D1(x1)
        # resnet block
        y = self.resnet_block(y)
        # output
        y = self.conv2Dout(tr.relu(y)).squeeze()

        y = y.multiply(mask)

        yt = tr.transpose(y, -1, -2)
        y = (y + yt) / 2

        return y, y0

    def loss_func(self, yhat, y):
        """yhat and y are [N, M, M]"""
        y = y.view(y.shape[0], -1)
        yhat, y0 = yhat  # yhat is the final ouput and y0 is the cnn output

        yhat = yhat.view(yhat.shape[0], -1)
        y0 = y0.view(y0.shape[0], -1)

        # Add l1 loss, ignoring the padding
        l1_loss = tr.mean(tr.relu(yhat[y != -1]))

        # yhat has to be shape [N, 2, L].
        yhat = yhat.unsqueeze(1)
        # yhat will have high positive values for base paired and high negative values for unpaired
        yhat = tr.cat((-yhat, yhat), dim=1)
        error_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)

        y0 = y0.unsqueeze(1)
        y0 = tr.cat((-y0, y0), dim=1)
        error_loss1 = cross_entropy(y0, y, ignore_index=-1, weight=self.class_weight)

        loss = (
            error_loss
            + self.loss_beta * error_loss1
            + self.loss_l1 * l1_loss
        )
        return loss

    def fit(self, loader):
        self.train()
        metrics = {"loss": 0, "f1": 0}

        if self.verbose:
            loader = tqdm(loader)

        for batch in loader:  # X, Y, L, mask, seqid, sequence, [prob_mask]
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)

            self.optimizer.zero_grad()  # Cleaning cache optimizer
            y_pred = self(X, *batch[2:])

            loss = self.loss_func(y_pred, y)

            # y_pred is a composed tensor, we need to get the final pred
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            f1 = contact_f1(
                y.cpu(), y_pred.detach().cpu(), batch[2], method="triangular"
            )

            metrics["loss"] += loss.item()
            metrics["f1"] += f1

            loss.backward()
            self.optimizer.step()

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def test(self, loader):
        self.eval()
        metrics = {"loss": 0, "f1": 0}

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:  
                X = batch[0].to(self.device)
                y = batch[1].to(self.device)
                lengths = batch[2]
                mask = batch[3]

                y_pred = self(X, *batch[2:])
                loss = self.loss_func(y_pred, y)
                metrics["loss"] += loss.item()

                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = postprocessing(y_pred.cpu(), mask)

                f1 = contact_f1(
                    y.cpu(), y_pred_post.cpu(), lengths, reduce=True)

                metrics["f1"] += f1

        for k in metrics:
            metrics[k] /= len(loader)

        if self.scheduler is not None:
            self.scheduler.step(metrics["f1"])

        return metrics

    def pred(self, loader):
        self.eval()

        if self.verbose:
            loader = tqdm(loader)

        predictions = []
        with tr.no_grad():
            for batch in loader: # X, Y, L, mask, seqid, sequence, [prob_mask]
                X = batch[0].to(self.device)
                lengths = batch[2]
                mask = batch[3]
                seqid = batch[4]
                sequences = batch[5]

                y_pred = self(X, *batch[2:])
                
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = postprocessing(y_pred.cpu(), mask)

                for k in range(y_pred_post.shape[0]):
                    predictions.append(
                        (seqid[k],
                         sequences[k],
                            mat2bp(
                                y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                            )                         
                        )
                    )
               
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])

        return predictions

class ResidualLayer1D(nn.Module):
    def __init__(
        self,
        dilation,
        resnet_bottleneck_factor,
        filters,
        kernel_size,
    ):
        super().__init__()

        num_bottleneck_units = math.floor(resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(
                filters,
                num_bottleneck_units,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(num_bottleneck_units),
            nn.ReLU(),
            nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"),
        )

    def forward(self, x):
        return x + self.layer(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, filters, filters1, kernel_size, dilation):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters1, kernel_size, padding="same"),
            nn.BatchNorm2d(filters1),
            nn.ReLU(),
            nn.Conv2d(
                filters1, filters, kernel_size, dilation=dilation, padding="same"
            ),
        )

    def forward(self, x):
        return self.layer(x) + x
