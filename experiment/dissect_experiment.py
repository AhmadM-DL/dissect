# New-style dissection experiment code.
import torch, argparse, os, shutil, inspect, json, numpy, random
from collections import defaultdict
from netdissect import pbar, nethook, renormalize, pidfile, zdataset
from netdissect import upsample, tally, imgviz, imgsave, bargraph
import os
import setting
import netdissect
torch.backends.cudnn.benchmark = True

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model',  default='alexnet')
    aa('--dataset', choices=['places', 'church', 'kitchen', 'livingroom',
                             'bedroom'],
            default='places')
    aa('--seg', choices=['net', 'netp', 'netq', 'netpq', 'netpqc', 'netpqxc'],
            default='netpqc')
    aa('--layer', default=None)
    aa('--quantile', type=float, default=0.01)
    aa('--miniou', type=float, default=0.04)
    aa('--thumbsize', type=int, default=100)
    aa('--model_path', type=str, default=None)
    aa("--output_dir", type=str, default="results")
    aa("--width_multiplier", type=int, default=1)
    aa("--sk_ratio", type=int, default=0)
    args = parser.parse_args()
    return args

def main(args):
    resdir =  os.path.join(args.output_dir, '%s-%s-%s' % (args.model, args.dataset, args.seg)) 
    if args.layer is not None:
        resdir += '-' + args.layer
    if args.quantile != 0.005:
        resdir += ('-%g' % (args.quantile * 1000))
    if args.thumbsize != 100:
        resdir += ('-t%d' % (args.thumbsize))
    resfile = pidfile.exclusive_dirfn(resdir)

    model = load_model(args)
    layername = instrumented_layername(args)
    model.retain_layer(layername)
    dataset = load_dataset(args, model=model.model)
    upfn = make_upfn(args, dataset, model, layername)
    sample_size = len(dataset)
    is_generator = (args.model == 'progan')
    percent_level = 1.0 - args.quantile
    iou_threshold = args.miniou
    image_row_width = 5

    # Tally rq.np (representation quantile, unconditional).
    pbar.descnext('rq')
    def compute_samples(batch, *args):
        data_batch = batch.cuda()
        _ = model(data_batch)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])
    rq = tally.tally_quantile(compute_samples, dataset,
                            sample_size=sample_size,
                            r=8192,
                            num_workers=100,
                            pin_memory=True,
                            cachefile=resfile('rq.npz'))

    # Create visualizations - first we need to know the topk
    pbar.descnext('topk')
    def compute_image_max(batch, *args):
        data_batch = batch.cuda()
        _ = model(data_batch)
        acts = model.retained_layer(layername)
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.max(2)[0]
        return acts
    topk = tally.tally_topk(compute_image_max, dataset, sample_size=sample_size,
            batch_size=50, num_workers=30, pin_memory=True,
            cachefile=resfile('topk.npz'))

    # Visualize top-activating patches of top-activatin images.
    pbar.descnext('unit_images')
    image_size, image_source = None, None
    if is_generator:
        image_size = model(dataset[0][0].cuda()[None,...]).shape[2:]
    else:
        image_source = dataset
    iv = imgviz.ImageVisualizer((args.thumbsize, args.thumbsize),
        image_size=image_size,
        source=dataset,
        quantiles=rq,
        level=rq.quantiles(percent_level))
    def compute_acts(data_batch, *ignored_class):
        data_batch = data_batch.cuda()
        out_batch = model(data_batch)
        acts_batch = model.retained_layer(layername)
        if is_generator:
            return (acts_batch, out_batch)
        else:
            return (acts_batch, data_batch)
    unit_images = iv.masked_images_for_topk(
            compute_acts, dataset, topk,
            k=image_row_width, num_workers=30, pin_memory=True,
            cachefile=resfile('top%dimages.npz' % image_row_width))
    pbar.descnext('saving images')
    imgsave.save_image_set(unit_images, resfile('image/unit%d.jpg'),
            sourcefile=resfile('top%dimages.npz' % image_row_width))

    # Compute IoU agreement between segmentation labels and every unit
    # Grab the 99th percentile, and tally conditional means at that level.
    level_at_99 = rq.quantiles(percent_level).cuda()[None,:,None,None]

    segmodel, seglabels, segcatlabels = setting.load_segmenter(args.seg)
    renorm = renormalize.renormalizer(dataset, target='zc')
    def compute_conditional_indicator(batch, *args):
        data_batch = batch.cuda()
        out_batch = model(data_batch)
        image_batch = out_batch if is_generator else renorm(data_batch)
        seg = segmodel.segment_batch(image_batch, downsample=4)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        iacts = (hacts > level_at_99).float() # indicator
        return tally.conditional_samples(iacts, seg)

    pbar.descnext('condi99')
    condi99 = tally.tally_conditional_mean(compute_conditional_indicator,
            dataset, sample_size=sample_size,
            num_workers=3, pin_memory=True,
            cachefile=resfile('condi99.npz'))

    # Now summarize the iou stats and graph the units
    iou_99 = tally.iou_from_conditional_indicator_mean(condi99)
    unit_label_99 = [
            (concept.item(), seglabels[concept],
                segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*iou_99.max(0))]
    labelcat_list = [labelcat
            for concept, label, labelcat, iou in unit_label_99
            if iou > iou_threshold]
    save_conceptcat_graph(resfile('concepts_99.svg'), labelcat_list)
    dump_json_file(resfile('report.json'), dict(
            header=dict(
                name='%s %s %s' % (args.model, args.dataset, args.seg),
                image='concepts_99.svg'),
            units=[
                dict(image='image/unit%d.jpg' % u,
                    unit=u, iou=iou, label=label, cat=labelcat[1])
                for u, (concept, label, labelcat, iou)
                in enumerate(unit_label_99)])
            )
    copy_static_file('report.html', resfile('+report.html'))
    resfile.done();

def make_upfn(args, dataset, model, layername):
    '''Creates an upsampling function.'''
    convs, data_shape = None, None
    if args.model == 'alexnet':
        convs = [layer for name, layer in model.model.named_children()
                if name.startswith('conv') or name.startswith('pool')]
    elif args.model == 'progan':
        # Probe the data shape
        out = model(dataset[0][0][None,...].cuda())
        data_shape = model.retained_layer(layername).shape[2:]
        upfn = upsample.upsampler(
                (64, 64),
                data_shape=data_shape,
                image_size=out.shape[2:])
        return upfn
    else:
        # Probe the data shape
        _ = model(dataset[0][0][None,...].cuda())
        data_shape = model.retained_layer(layername).shape[2:]
        pbar.print('upsampling from data_shape', tuple(data_shape))
    upfn = upsample.upsampler(
            (56, 56),
            data_shape=data_shape,
            source=dataset,
            convolutions=convs)
    return upfn

def instrumented_layername(args):
    '''Chooses the layer name to dissect.'''
    if args.layer is not None:
        if args.model == 'vgg16':
            return 'features.' + args.layer
        elif args.model == "dc_vgg16":
            return 'features.' + args.layer
        elif args.model == "dc_alexnet":
            return 'features.'+ args.layer
        elif args.model == "m_dc_alexnet":
            return 'features.'+ args.layer
        elif args.model == "infomax_resnet50":
            return "encoder." + args.layer
        return args.layer
    # Default layers to probe
    if args.model == 'alexnet':
        return 'conv5'
    elif args.model == 'vgg16':
        return 'features.conv5_3'
    elif args.model == 'resnet152':
        return '7'
    elif args.model == 'progan':
        return 'layer4'

def load_model(args):
    '''Loads one of the benchmark classifiers or generators.'''
    if args.model in ['alexnet', 'vgg16', 'resnet152']:
        model = setting.load_classifier(args.model)
    elif args.model == 'progan':
        model = setting.load_proggan(args.dataset)

    elif "insdis" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url = args.model_path
        else:
            url= 'https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet50_update.pth'
        model = setting.load_insdis_models(arch, url)

    elif "swav" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar'
        model = setting.load_swav_models(arch, url)
    
    elif "pcl_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v1_epoch200.pth.tar'
        model = setting.load_pcl_models(arch, url)

    elif "pclv2_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar'
        model = setting.load_pcl_models(arch, url)

    elif "random" in args.model:
        arch = args.model.split("_")[1]
        model = setting.load_random_models(arch)

    elif "supervised" in args.model:
        arch = args.model.split("_")[1]
        model = setting.load_supervised_models(arch)

    elif "jigsaw" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= ''
        model = setting.load_pcl_models(arch, url)

    elif "simCLR" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            print("SimCLR requiers a local checkpoint path")
        model = setting.load_simclr_models(arch, url, args.width_multiplier, args.sk_ratio)

    elif "selav2_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar'
        model = setting.load_sela_v2_models(arch, url) 
    
    elif "sela_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'http://www.robots.ox.ac.uk/~vgg/research/self-label/asset/new_models/resnet50-10x3k_pp.pth'
        model = setting.load_sela_models(arch, url)    

    elif "mocov2_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar'
        model = setting.load_moco_models(arch, url)
    
    elif "moco_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar'
        model = setting.load_moco_models(arch, url)
        
    elif "npid" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet50_update.pth'
        model = setting.load_npid_models(arch, url)        
    elif "m_dc_" in args.model:
        arch = args.model.split("_")[2]
        model = setting.load_m_deep_cluster_models(arch, args.model_path)

    elif "dc_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://dl.fbaipublicfiles.com/deepcluster/'+arch+'/checkpoint.pth.tar'
        model = setting.load_deep_cluster_models(arch, url)

    elif "dcv2_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar'
        model = setting.load_deep_cluster_v2_models(arch, url)
    
    elif "infomax_" in args.model:
        arch = args.model.split("_")[1]
        if args.model_path:
            url= args.model_path
        else:
            url= 'https://uc66f9ba96803815b4e03bedd473.dl.dropboxusercontent.com/cd/0/get/BIT9LOurKc7ynGIi9_lv6CPoDF-NLW4-NDc_MqePyVAY8QQh2s3Yz6Zg9WJCNa_CYKpi4ge7u9BTvLZTH3tFxQ9gX4ejTEQVUNdAR1Ie1XBmtWTnKWiaPKuWmSBhJTFIe6s/file?_download_id=019700423140138645822466014310099868483434900982898758624353033122&_notify_domain=www.dropbox.com&dl=1'
        model = setting.load_infomax_models(arch, url)

    model = nethook.InstrumentedModel(model).cuda().eval()
    return model

def load_dataset(args, model=None):
    '''Loads an input dataset for testing.'''
    from torchvision import transforms
    if args.model == 'progan':
        dataset = zdataset.z_dataset_for_model(model, size=10000, seed=1)
        return dataset
    elif args.dataset in ['places']:
        crop_size = 227 if args.model == 'alexnet' else 224
        return setting.load_dataset(args.dataset, split='val', full=True,
                crop_size=crop_size, download=True)
    assert False

def graph_conceptcatlist(conceptcatlist, **kwargs):
    count = defaultdict(int)
    catcount = defaultdict(int)
    for c in conceptcatlist:
        count[c] += 1
    for c in count.keys():
        catcount[c[1]] += 1
    cats = ['object', 'part', 'material', 'texture', 'color']
    catorder = dict((c, i) for i, c in enumerate(cats))
    sorted_labels = sorted(count.keys(),
        key=lambda x: (catorder[x[1]], -count[x]))
    sorted_labels
    return bargraph.make_svg_bargraph(
        [label for label, cat in sorted_labels],
        [count[k] for k in sorted_labels],
        [(c, catcount[c]) for c in cats], **kwargs)

def save_conceptcat_graph(filename, conceptcatlist):
    svg = graph_conceptcatlist(conceptcatlist, barheight=80, file_header=True)
    with open(filename, 'w') as f:
        f.write(svg)


class FloatEncoder(json.JSONEncoder):
    def __init__(self, nan_str='"NaN"', **kwargs):
        super(FloatEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def iterencode(self, o, _one_shot=False):
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring
        def floatstr(o, allow_nan=self.allow_nan,
                _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY,
                nan_str=self.nan_str):
            if o != o:
                text = nan_str
            elif o == _inf:
                text = '"Infinity"'
            elif o == _neginf:
                text = '"-Infinity"'
            else:
                return repr(o)
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))
            return text

        _iterencode = json.encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)

def dump_json_file(target, data):
    with open(target, 'w') as f:
        json.dump(data, f, indent=1, cls=FloatEncoder)

def copy_static_file(source, target):
    sourcefile = os.path.join(
            os.path.dirname(inspect.getfile(netdissect)), source)
    shutil.copy(sourcefile, target)

if __name__ == '__main__':
    args = parseargs()
    main(args)

