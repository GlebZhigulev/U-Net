import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds = torch.sigmoid(preds)
        smooth = 1e-5
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (
            preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
        )
        dice_loss = 1 - dice.mean()

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
