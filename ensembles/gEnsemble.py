from experiment.dissect_experiment import load_model
from torchvision.datasets.imagenet import ImageNet
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
import logging
import sys

MODELS = ["", "", ""]
SEPERATOR = "|"


def get_features_model(model, device):
    if "swav_resnet50" == model.lower():
        class args:
            model = "swav_resnet50"
            model_path = None
        model = load_model(args).model
        model = torch.nn.Sequential(*(list(model.children())[:-2]))
        output_size = model(torch.rand(
            (1, 3, 244, 244), device=device)).shape[1]
        return model, output_size
    if "supervised_resnet50" == model.lower():
        class args:
            model = "supervised_resnet50"
            model_path = None
        model = load_model(args).model
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        output_size = model(torch.rand(
            (1, 3, 244, 244), device=device)).shape[1]
        return model, output_size


class Ensemble2(torch.nn.Module):
    def __init__(self, m1, m2, fc_layers, device, dropout_rate=None):
        super(Ensemble2, self).__init__()
        self.model_1 = m1
        self.model_2 = m2

        self._freeze(m1)
        self._freeze(m2)

        self.model_1.to(device)
        self.model_2.to(device)

        self.classifier = self._classifier_from_list(
            fc_layers, dropout_rate=dropout_rate)
        self.classifier.to(device)
        self.device = device

    def forward(self, x):
        x1 = self.model_1(x)
        x2 = self.model_2(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _classifier_from_list(self, layers_sizes, dropout_rate=0.5):
        fc_layers = []0
        for s1, s2 in zip(layers_sizes, layers_sizes[1:]):
            if dropout_rate:
                fc.layers.append(torch.nn.Dropout(dropout_rate))
            fc_layers.append(torch.nn.Linear(s1, s2))
        return torch.nn.Sequential(*fc_layers)

    def train_(self, optimizer, train_dataloader, loss_fn):
        self.train()
        losses = []
        for i, (input, target) in enumerate(train_dataloader):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self(input)

            loss = loss_fn(output, target)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.average(losses)

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def val(self, val_dataloader, loss_fn):
        self.eval()
        accuracies_1 = []
        accuracies_5 = []
        losses = []
        for i, (input, target) in enumerate(val_dataloader):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self(input)

            # measure accuracy and record loss
            acc1, acc5 = self._accuracy(output.data, target, topk=(1, 5))
            loss = loss_fn(output, target)

            accuracies_1.append(acc1.item())
            accuracies_5.append(acc5.item())
            losses.append(loss.item())

        return np.average(accuracies_1), np.average(accuracies_5), np.average(losses)


def main(args):
    models = args.models.split(SEPERATOR)
    logging.info("Downloading models")
    models = [get_features_model(m) for m in models]
    logging.info("Building Ensemble")
    if len(models) == 2:
        fc_layers = [models[0][1]+models[0][1], 1000]
        ensemble = Ensemble2(
            models[0][0], models[1][0], fc_layers, args.device, args.dropout_rate)
    else:
        raise Exception("Such number of models is not supported yet")

    logging.info("Initializing Imagenet Dataset")
    imagenet_train = ImageNet(root=args.imagenet, split="train")
    imagenet_valid = ImageNet(root=args.imagenet, split="val")

    logging.info("Initializing Dataloaders")
    train_dataloader = torch.utils.data.DataLoader(
        imagenet_train, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(
        imagenet_valid, batch_size=args.batch_size, shuffle=False)

    logging.info("Initializing Optimizer (lr: %f, mom:%f, wd:%f)"%(args.lr, args.momentum, args.wd))
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, ensemble.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    # Define experiment tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir+"/"+args.models)
    logging.info("Writing training logs to %s"%writer)

    # Train/Validate on Imagenet
    logging.info("Start Training for %d epochs"%args.n_epochs)
    for i in range(args.n_epochs):
        logging.info("Epoch %d: " % i)
        loss = ensemble.train_(optimizer, train_dataloader,
                               torch.nn.CrossEntropyLoss())
        acc1, acc5, valid_loss = ensemble.val(
            valid_dataloader, torch.nn.CrossEntropyLoss())

        writer.add_scalars('loss', {'train': loss, 'valid': valid_loss}, i)
        writer.add_scalars('acc/acc1', acc1, i)
        writer.add_scalars('acc/acc5', acc2, i)


if __name__ == '__main__':
    description =
    """
    A module to merge features of multiple unsupervised deep learning models.
    The modile takes a list of strings representing the models to merge.
    The module will train a FC classifier on top of the features on Imagenet
    and report accuracy.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--models", type=str, help="A List of module in a string format as in m1|m2|m3|... ")
    parser.add_argument("--device", type=str,
                        help="Device to run computation on", default="cuda:0")
    parser.add_argument("--imagenet", type=str,
                        help="Path to imagenet dataset root")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning Rate")
    parser.add_argument("--momentum", type=float,
                        default=0.9, help="Optimizer Momentum")
    parser.add_argument("--wd", type=float, default=0.0001,
                        help="Optimizer Weight Decay")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Training Batch Size")
    parser.add_argument("--log_dir", type=str, default="./runs",
                        help="Tensorboard logs directory")
    parser.add_argument("--dropout_rate", type=float,
                        default=0.5, help="Add dropout to classifier")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    main(args)
