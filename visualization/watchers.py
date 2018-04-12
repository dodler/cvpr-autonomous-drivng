import visdom
import numpy as np

vis = visdom.Visdom()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomValueWatcher(object):
    def __init__(self):
        self._watchers = {}
        self._wins = {}
        self._vis = visdom.Visdom()
        self.vis_img = None
        self.vis_gt = None
        self.vis_pred = None

    def display_every_iter(self, iter_num, X, gt, prediction, base_label):
        if iter_num % 10 == 0:
            img = X.data.squeeze(0).cpu().numpy()[0]
            #                img = np.transpose(img, axes=[1, 2, 0])
            mask = gt.data.squeeze(0).cpu().numpy()[0]
            pred = (prediction > 0.6).float().data.squeeze(0).cpu().numpy()[0]
            #                Q = dense_crf(((img*255).round()).astype(np.uint8), pred)

            # yy = dense_crf(np.array(prediction).astype(np.uint8), y)

            if self.vis_img is None:
                vis.image(img, opts=dict(title='source image'))
            else:
                vis.image(img, opts=dict(title='source image'), win=self.vis_img)

            if self.vis_gt is None:
                self.vis_gt = self._vis.image(mask, opts=dict(title='gt'))
            else:
                self._vis.image(img, opts=dict(title='gt'), win=self.vis_gt)

            if self.vis_gt is None:
                self.vis_pred = self._vis.image(pred, opts=dict(title='prediction'))
            else:
                self._vis.image(img, opts=dict(title='prediction'), win=self.vis_pred)

    def get_vis(self):
        return self._vis

    def add_value(self, name, value):
        if name in self._watchers.keys():
            self._watchers[name].append(value)
        else:
            self._watchers[name] = [value]

    def output(self):
        for name in self._wins.keys():
            self._vis.line(Y=np.array(self._watchers[name]),
                           X=np.array(range(len(self._watchers[name]))),
                           win=self._wins[name], update='new',
                           opts=dict(title=name))

    def output(self, name):
        if name in self._wins.keys():
            self._vis.line(Y=np.array(self._watchers[name]),
                           X=np.array(range(len(self._watchers[name]))),
                           win=self._wins[name], update='new',
                           opts=dict(title=name))
        else:
            self._wins[name] = self._vis.line(Y=np.array(self._watchers[name]),
                                              X=np.array(range(len(self._watchers[name]))),
                                              opts=dict(title=name))

    def clean(self, name):
        self._watchers[name] = []
