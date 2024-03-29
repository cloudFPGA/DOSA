#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#
import torch
import torch.nn as nn
from torch.autograd import Function


class SquaredHingeLoss(Function):
    @staticmethod
    def transform_target(targets, predictions):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target_onehot = torch.Tensor(targets.size(0), predictions.size(1)).to(device)
        target_onehot.fill_(-1)
        target_onehot.scatter_(1, targets.unsqueeze(1), 1)
        return target_onehot

    @staticmethod
    def forward(ctx, predictions, targets):
        target_onehot = SquaredHingeLoss.transform_target(targets, predictions)
        ctx.save_for_backward(predictions, target_onehot)
        output = 1.0 - predictions.mul(target_onehot)
        output[output.le(0.0)] = 0.0
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, *grad_output):
        predictions, targets = ctx.saved_tensors
        output = 1.0 - predictions.mul(targets)
        output[output.le(0.0)] = 0.0
        grad_output = grad_output[0]
        grad_output.resize_as_(predictions).copy_(targets).mul_(-2.0).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(predictions.numel())
        return grad_output, None


class SqrHingeLoss(nn.Module):
    # Squared Hinge Loss
    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        return SquaredHingeLoss.apply(input, target)
