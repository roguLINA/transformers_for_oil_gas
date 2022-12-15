"""Model architectures."""

from typing import Tuple, Dict, List, Any

import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trans_oil_gas import transformer, informer, performer

from trans_oil_gas.utils_dataset import TripletDatasetSlices
from trans_oil_gas.utils_model_training import calculate_metrics

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)


class SiameseArchitecture(nn.Module):
    """Implements a Siamese Neural Network based on transformer block."""

    def __init__(
        self,
        encoder_type: str,
        fc_hidden_size: int,
        fc_output_size: int = 1,
        output_transform: str = "sigmoid",
        **specific_encoder_params: Dict[str, Any],
    ) -> None:
        """Initialize model with Siamese Architecture.

        :param encoder_type: type of transformer-based encoder ('transformer', 'informer', and 'performer' are available)
        :param fc_hidden_size: hidden size in FC-part
        :param fc_output_size: size of Siamese model's output
        :param output_transform: output transformation ('sigmoid' and 'identity' are available)
        :param specific_encoder_params: specific parameters for each encoder type
        """
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "transformer":
            # Positional encoding for sequences
            self.positional_encoding = transformer.PositionalEncoding(
                d_model=specific_encoder_params["d_model"]
            )

            # transformer part
            self.encoder = transformer.TransformerEncoder(
                num_layers=specific_encoder_params["num_layers"],
                input_dim=specific_encoder_params["d_model"],
                num_heads=specific_encoder_params["nhead"],
                dim_feedforward=specific_encoder_params["dim_feedforward"],
                dropout=specific_encoder_params["dropout"],
            )
            in_emb = specific_encoder_params["d_model"] * 100
            out_emb = fc_hidden_size  

        elif encoder_type == "informer":
            self.positional_encoding = None
            self.encoder = informer.InformerEncoder(
                enc_in=specific_encoder_params["enc_in"],
                factor=specific_encoder_params["factor"],
                d_model=specific_encoder_params["d_model"],
                n_heads=specific_encoder_params["n_heads"],
                e_layers=specific_encoder_params["e_layers"],
                d_ff=specific_encoder_params["d_ff"],
                dropout=specific_encoder_params["dropout"],
                attn=specific_encoder_params["attn"],
                activation=specific_encoder_params["activation"],
                output_attention=specific_encoder_params["output_attention"],
                distil=specific_encoder_params["distil"],
                device=specific_encoder_params["device"],
            )
            in_emb = (
                specific_encoder_params["d_model"] * specific_encoder_params["n_seq"]
            )
            out_emb = specific_encoder_params["d_model"]

        elif encoder_type == "performer":
            self.positional_encoding = None
            self.encoder = performer.PerformerEncoder(
                dim=specific_encoder_params["dim"],
                head_num=specific_encoder_params["head_num"],
                dropout=specific_encoder_params["dropout"],
                nb_random_features=specific_encoder_params["nb_random_features"],
                use_relu_kernel=specific_encoder_params["use_relu_kernel"],
                device=specific_encoder_params["device"],
            )
            in_emb = specific_encoder_params["dim"] * specific_encoder_params["n_seq"]
            out_emb = specific_encoder_params["dim"]

        # layer to change size of transformer output to desired embedded size
        self.embed_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_emb, out_emb),
        )

        # 3 fully-connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * out_emb, out_features=fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=specific_encoder_params["dropout"]),
            nn.Linear(in_features=fc_hidden_size, out_features=fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=specific_encoder_params["dropout"]),
            nn.Linear(in_features=fc_hidden_size, out_features=fc_output_size),
        )

        # activation to get probs
        self.tfm = nn.Sigmoid() if output_transform == "sigmoid" else nn.Identity()

    def encode(
        self, input_slice: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Get transformer-based model embedding.

        :param input_slice: input sequence
        :param mask: mask
        :return: sequence's embedding
        """
        x = self.encoder(input_slice)
        x = self.embed_layer(x)

        return x

    def forward(
        self,
        input_slices: List[torch.Tensor],
        mask: torch.Tensor = None,
        add_positional_encoding: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        :param input_slices: input sequences
        :param mask: mask
        :param add_positional_encoding: indicate if apply positional encoding
        :return: probability if two input slices belong to the same well
        """
        slice_1 = input_slices[0]
        slice_2 = input_slices[1]

        if add_positional_encoding and self.positional_encoding is not None:
            slice_1 = self.positional_encoding(slice_1)
            slice_2 = self.positional_encoding(slice_2)

        embedding_out_1 = self.encode(slice_1, mask=mask)
        embedding_out_2 = self.encode(slice_2, mask=mask)

        embedding_out = torch.cat([embedding_out_1, embedding_out_2], dim=1)

        return self.tfm(self.fc(embedding_out))

    @torch.no_grad()
    def get_attention_maps(
        self,
        x: torch.tensor,
        mask: torch.tensor = None,
        add_positional_encoding: bool = True,
    ) -> torch.tensor:
        """Extract the attention matrices of the whole Transformer for a single batch. Input arguments same as the forward pass.

        :param x: input sequences
        :param mask: mask
        :param add_positional_encoding: indicate if apply positional encoding
        :return: attention matrix
        """
        if self.encoder_type == "transformer":
            if add_positional_encoding:
                x = self.positional_encoding(x)
            attention_maps = self.encoder.get_attention_maps(x, mask=mask)
            return attention_maps


class TripletArchitecture(nn.Module):
    """Implements a Triplet Neural Network based on transformer block."""

    def __init__(
        self, encoder_type: str, **specific_encoder_params: Dict[str, Any]
    ) -> None:
        """Initialize model with Triplet Architecture.

        :param encoder_type: type of transformer-based encoder ('transformer', 'informer', and 'performer' are available)
        :param specific_encoder_params: specific parameters for each encoder type
        """
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "transformer":
            # Positional encoding for sequences
            self.positional_encoding = transformer.PositionalEncoding(
                d_model=specific_encoder_params["d_model"]
            )

            # transformer part
            self.encoder = transformer.TransformerEncoder(
                num_layers=specific_encoder_params["num_layers"],
                input_dim=specific_encoder_params["d_model"],
                num_heads=specific_encoder_params["nhead"],
                dim_feedforward=specific_encoder_params["dim_feedforward"],
                dropout=specific_encoder_params["dropout"],
            )
            in_emb = specific_encoder_params["d_model"] * 100
            out_emb = (
                specific_encoder_params["embedding_size"]
                if "embedding_size" in specific_encoder_params.keys()
                else specific_encoder_params["hidden_size"]
            )

        elif encoder_type == "informer":
            self.positional_encoding = None
            self.encoder = informer.InformerEncoder(
                enc_in=specific_encoder_params["enc_in"],
                factor=specific_encoder_params["factor"],
                d_model=specific_encoder_params["d_model"],
                n_heads=specific_encoder_params["n_heads"],
                e_layers=specific_encoder_params["e_layers"],
                d_ff=specific_encoder_params["d_ff"],
                dropout=specific_encoder_params["dropout"],
                attn=specific_encoder_params["attn"],
                activation=specific_encoder_params["activation"],
                output_attention=specific_encoder_params["output_attention"],
                distil=specific_encoder_params["distil"],
                device=specific_encoder_params["device"],
            ).double()
            in_emb = (
                specific_encoder_params["d_model"] * specific_encoder_params["n_seq"]
            )
            out_emb = specific_encoder_params["d_model"]

        elif encoder_type == "performer":
            self.positional_encoding = None
            self.encoder = performer.PerformerEncoder(
                dim=specific_encoder_params["dim"],
                head_num=specific_encoder_params["head_num"],
                dropout=specific_encoder_params["dropout"],
                nb_random_features=specific_encoder_params["nb_random_features"],
                use_relu_kernel=specific_encoder_params["use_relu_kernel"],
                device=specific_encoder_params["device"],
            ).double()
            in_emb = specific_encoder_params["dim"] * specific_encoder_params["n_seq"]
            out_emb = specific_encoder_params["dim"]

        self.embed_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_emb, out_emb),
        )

    def encode(
        self, input_slice: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Get transformer-based model embedding.

        :param input_slice: input sequence
        :param mask: mask
        :return: sequence's embedding
        """
        x = self.encoder(input_slice)
        x = self.embed_layer(x)

        return x

    def forward(
        self,
        x: List[torch.Tensor],
        mask: torch.Tensor = None,
        add_positional_encoding: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        :param input_slices: input sequences
        :param mask: mask
        :param add_positional_encoding: indicate if apply positional encoding
        :return: probability if two input slices belong to the same well
        """
        if add_positional_encoding and self.encoder_type == "transformer":
            x = self.positional_encoding(x)

        return self.encode(x, mask=mask)

    @torch.no_grad()
    def get_attention_maps(
        self,
        x: torch.tensor,
        mask: torch.tensor = None,
        add_positional_encoding: bool = True,
    ) -> torch.tensor:
        """Extract the attention matrices of the whole Transformer for a single batch. Input arguments same as the forward pass.

        :param x: input sequences
        :param mask: mask
        :param add_positional_encoding: indicate if apply positional encoding
        :return: attention matrix
        """
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.encoder.get_attention_maps(x, mask=mask)
        return attention_maps


# ------------------------- Similarity models (Siamese and Triplet) -------------------------


class IntervalModel(pl.LightningModule):
    """PyTorch Lightning wrapper for convenient training models."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        train_data: Dataset,
        test_data: Dataset,
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        """Initialize PyTorch Wrapper.

        :param model_type: model type in format <siamese/triplet>_<informer/transformer/performer>
        :param train_data: Dataset with training data
        :param test_data: Dataset with test data
        :param batch_size: batch size for DataLoader
        :param learning_rate: hyperparameter that controls how much to change the model in response to
            the estimated error each time the model weights are updated.
        """
        super(IntervalModel, self).__init__()
        self.model = model
        self.model_type = model_type

        self.train_data = train_data
        self.val_data = test_data

        self.batch_size = batch_size
        self.learning_rate = 1e-3 if "siamese" in self.model_type else 1e-4
        self.kwargs = kwargs

        if "siamese" in model_type:
            self.loss_function = nn.BCELoss()

        elif "triplet" in model_type:
            self.loss_function = nn.TripletMarginLoss(margin=1.75)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get model's output.

        :param inputs: input data for model
        :return: base model's data
        """
        return self.model(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Do training step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: loss on training data
        """
        anchor, positive, negative, well_anchor = batch
        if "siamese" in self.model_type:
            target_1_pred = self.forward([anchor, positive])
            target_0_pred = self.forward([anchor, negative])

            predictions = torch.cat((target_1_pred.squeeze(), target_0_pred.squeeze()))
            all_targets = torch.cat(
                (
                    torch.ones(anchor.shape[0]).float(),
                    torch.zeros(anchor.shape[0]).float(),
                )
            ).to(predictions.device)

            train_loss = self.loss_function(predictions, all_targets)
            train_accuracy = (
                (all_targets == (predictions.squeeze() > 0.5)).float().mean()
            )

            self.log(
                "train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                "train_acc", train_accuracy, on_step=False, on_epoch=True, prog_bar=True
            )
        else:
            anchor_out = self.forward(anchor)
            positive_out = self.forward(positive)
            negative_out = self.forward(negative)

            train_loss = self.loss_function(anchor_out, positive_out, negative_out)

            self.log(
                "train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True
            )

        return train_loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do validation step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: dict with loss, accuracy, target and predictions
        """
        anchor, positive, negative, well_anchor = batch
        if "siamese" in self.model_type:
            target_1_pred = self.forward([anchor, positive])
            target_0_pred = self.forward([anchor, negative])

            predictions = torch.cat((target_1_pred.squeeze(), target_0_pred.squeeze()))
            all_targets = torch.cat(
                (
                    torch.ones(anchor.shape[0]).float(),
                    torch.zeros(anchor.shape[0]).float(),
                )
            ).to(predictions.device)

            val_loss = self.loss_function(predictions, all_targets)
            val_accuracy = (all_targets == (predictions.squeeze() > 0.5)).float().mean()

            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_acc", val_accuracy, prog_bar=True)
            return {
                "val_loss": val_loss,
                "val_acc": val_accuracy,
                "val_target": all_targets,
                "val_predictions": predictions,
            }

        else:
            anchor_out = self.forward(anchor)
            positive_out = self.forward(positive)
            negative_out = self.forward(negative)

            val_loss = self.loss_function(anchor_out, positive_out, negative_out)

            self.log("val_loss", val_loss, prog_bar=True)

            return val_loss

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        """Calculate accuracy, ROC_AUC, PR_AUC after epoch end.

        :param outputs: dict with loss, accuracy, target and predictions
        """
        if "siamese" in self.model_type:
            predictions = torch.cat([x["val_predictions"] for x in outputs])
            target = torch.cat([x["val_target"] for x in outputs])

            accuracy, roc_auc, pr_auc, conf_matrix = calculate_metrics(
                target.detach().cpu().numpy().squeeze(),
                predictions.detach().cpu().numpy().squeeze(),
            )

            log_dict = {
                "mean_accuracy": accuracy,
                "mean_roc_auc": roc_auc,
                "mean_pr_auc": pr_auc,
            }

            for k, v in log_dict.items():
                self.log(k, v, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set parameters for optimizer.

        :return: optimizer
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Set DataLoader for training data.

        :return: DataLoader for training data
        """
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        """Set DataLoader for training data.

        :return: DataLoader for training data
        """
        val_dataloader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        return val_dataloader
