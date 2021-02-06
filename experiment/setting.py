import torch, torchvision, os, collections
from netdissect import parallelfolder, zdataset, renormalize, segmenter
import oldalexnet, oldvgg16, oldresnet152
import deep_cluster_models
import deep_cluster_m_models
import ssmodels
import collections
import torchvision.models as models
from collections import OrderedDict

def load_proggan(domain):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)
    from . import proggan
    weights_filename = dict(
        bedroom='proggan_bedroom-d8a89ff1.pth',
        church='proggan_churchoutdoor-7e701dd5.pth',
        conferenceroom='proggan_conferenceroom-21e85882.pth',
        diningroom='proggan_diningroom-3aa0ab80.pth',
        kitchen='proggan_kitchen-67f1e16c.pth',
        livingroom='proggan_livingroom-5ef336dd.pth',
        restaurant='proggan_restaurant-b8578299.pth',
        celebhq='proggan_celebhq-620d161c.pth')[domain]
    # Posted here.
    url = 'https://dissect.csail.mit.edu/models/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1+
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = proggan.from_state_dict(sd)
    return model

def adapt_state_to_model(state_dict, mapping):
    return

def load_m_deep_cluster_models(architecture, url):
    
    # local url
    if architecture == "alexnet":
        model = deep_cluster_m_models.AlexNet_Small(sobel=True, batch_normalization=True, device=torch.device("cpu"))
    
    model.load_model_parameters(url)

    if architecture == "vgg16":
        model.features = torch.nn.Sequential(collections.OrderedDict(zip([
            'conv1_1', 'batch_norm1_1', 'relu1_1',
            'conv1_2', 'batch_norm1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'batch_norm2_1', 'relu2_1',
            'conv2_2', 'batch_norm2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'batch_norm3_1', 'relu3_1',
            'conv3_2', 'batch_norm3_2', 'relu3_2',
            'conv3_3', 'batch_norm3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'batch_norm4_1', 'relu4_1',
            'conv4_2', 'batch_norm4_2', 'relu4_2',
            'conv4_3', 'batch_norm4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'batch_norm4_1', 'relu5_1',
            'conv5_2', 'batch_norm4_2', 'relu5_2',
            'conv5_3', 'batch_norm4_3', 'relu5_3',
            'pool5'],
            model.features)))
    elif architecture=="alexnet":
        model.features = torch.nn.Sequential(collections.OrderedDict(zip([
            'conv1', 'batch_norm1', 'relu1', "pool1",
            'conv2', 'batch_norm2', 'relu2', "pool2",
            'conv3', 'batch_norm3', 'relu3', 
            'conv4', 'batch_norm4', 'relu4',
            'conv5', 'batch_norm5', 'relu5', "pool3"
            ],
            model.features)))
    model.eval()
    return model

def load_random_models(architecture):
    model = models.__dict__[architecture]()
    model.eval()
    return model

def load_supervised_models(architecture):
    model = models.__dict__[architecture](pretrained=True)
    model.eval()
    return model

def load_npid_models(architecture, url):
    
    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)
    
    model = ssmodels.npid.__dict__[architecture]()

    # deal with a dataparallel table
    def strip_module(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('module.'))
    sd = sd["state_dict"]
    sd = {strip_module(key): val for key, val in sd.items()}
    model.load_state_dict(sd) 
    model.eval()
    return model

def load_moco_models(architecture, url):
    #ToDo
    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)

    model = models.__dict__[architecture]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    # rename moco pre-trained keys
    state_dict = sd['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    model.eval()
    return model

def load_pcl_models(architecture, url):
    #TODO

    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)

    model = models.__dict__[architecture]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    # rename moco pre-trained keys
    state_dict = sd['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    model.eval()
    return model

def load_sela_models(architecture, url):
    
    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)
    
    model = models.__dict__[architecture](pretrained=False, num_classes=3000)

    # deal with a dataparallel table
    def strip_module(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('module.'))
    sd = sd["state_dict"]
    sd = {strip_module(key): val for key, val in sd.items()}
    model.load_state_dict(sd) 
    model.eval()
    return model

def load_swav_models(architecture, url):

    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)
    
    model = ssmodels.swav.__dict__[architecture](
        normalize=True,
        hidden_mlp=2048,
        output_dim=128,
        nmb_prototypes=3000,
    )

    # deal with a dataparallel table
    def strip_module(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('module.'))

    sd = {strip_module(key): val for key, val in sd.items()}
    model.load_state_dict(sd) 
    model.eval()
    return model

class infomax_opt:
    arch = "resnet50"
    method = "InfoMin"
    modal = "RGB"
    jigsaw = True
    mem = "moco"
    aug = 'D' 
    head = 'mlp'
    nce_t =  0.15
    feat_dim = 128

def load_infomax_models(architecture, url):
    modal = "RGB"
    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)
    sd = sd["model"]

    model, _ = ssmodels.infomax.build_backbone.build_model(infomax_opt)

    if modal == 'RGB':
        # Unimodal (RGB) case
        encoder_state_dict = OrderedDict()
        for k, v in sd.items():
            k = k.replace('module.', '')
            if 'encoder' in k:
                k = k.replace('encoder.', '')
                encoder_state_dict[k] = v
        model.encoder.load_state_dict(encoder_state_dict)
    else:
        # Multimodal (CMC) case
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        for k, v in sd.items():
            k = k.replace('module.', '')
            if 'encoder1' in k:
                k = k.replace('encoder1.', '')
                encoder1_state_dict[k] = v
            if 'encoder2' in k:
                k = k.replace('encoder2.', '')
                encoder2_state_dict[k] = v
        model.encoder1.load_state_dict(encoder1_state_dict)
        model.encoder2.load_state_dict(encoder2_state_dict)

    return model

def load_sela_v2_models(architecture, url):
    
    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)
    
    model = ssmodels.swav.__dict__[architecture](
        normalize=True,
        hidden_mlp=2048,
        output_dim=128,
        nmb_prototypes=[3000, 3000, 3000]
    )

    # deal with a dataparallel table
    def strip_module(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('module.'))

    sd = {strip_module(key): val for key, val in sd.items()}
    model.load_state_dict(sd) 
    model.eval()
    return model

def load_deep_cluster_v2_models(architecture, url):

    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)
    
    model = ssmodels.swav.__dict__[architecture](
        normalize=True,
        hidden_mlp=2048,
        output_dim=128,
        nmb_prototypes=[3000,3000,3000],
    )

    # deal with a dataparallel table
    def strip_module(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('module.'))

    sd = {strip_module(key): val for key, val in sd.items()}
    model.load_state_dict(sd) 
    model.eval()
    return model

def load_deep_cluster_models(architecture, url):

    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
        #local url
        sd = torch.load(url)

    # size of the top layer
    N = sd['state_dict']['top_layer.bias'].size()

    # build skeleton of the model
    sob = 'sobel.0.weight' in sd['state_dict'].keys()
    model = deep_cluster_models.__dict__[architecture](sobel=sob, out=int(N[0]))

    # deal with a dataparallel table
    def rename_key(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('.module'))

    sd['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in sd['state_dict'].items()}

    # load weights
    model.load_state_dict(sd['state_dict'])
    if architecture == "vgg16":
        model.features = torch.nn.Sequential(collections.OrderedDict(zip([
            'conv1_1', 'batch_norm1_1', 'relu1_1',
            'conv1_2', 'batch_norm1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'batch_norm2_1', 'relu2_1',
            'conv2_2', 'batch_norm2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'batch_norm3_1', 'relu3_1',
            'conv3_2', 'batch_norm3_2', 'relu3_2',
            'conv3_3', 'batch_norm3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'batch_norm4_1', 'relu4_1',
            'conv4_2', 'batch_norm4_2', 'relu4_2',
            'conv4_3', 'batch_norm4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'batch_norm4_1', 'relu5_1',
            'conv5_2', 'batch_norm4_2', 'relu5_2',
            'conv5_3', 'batch_norm4_3', 'relu5_3',
            'pool5'],
            model.features)))
    elif architecture=="alexnet":
        model.features = torch.nn.Sequential(collections.OrderedDict(zip([
            'conv1', 'batch_norm1', 'relu1', "pool1",
            'conv2', 'batch_norm2', 'relu2', "pool2",
            'conv3', 'batch_norm3', 'relu3', 
            'conv4', 'batch_norm4', 'relu4',
            'conv5', 'batch_norm5', 'relu5', "pool3"
            ],
            model.features)))
    model.eval()
    return model

def load_insdis_models(architecture, url):
    if "http" in url:
        # remote url
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    else:
    #local url
        sd = torch.load(url)
    
    model = ssmodels.insdis.__dict__[architecture](low_dim=128)

    # deal with a dataparallel table
    def strip_module(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('module.'))

    sd = {strip_module(key): val for key, val in sd.items()}
    model.load_state_dict(sd) 
    model.eval()
    return model
    
def load_classifier(architecture):
    model_factory = dict(
            alexnet=oldalexnet.AlexNet,
            vgg16=oldvgg16.vgg16,
            resnet152=oldresnet152.OldResNet152)[architecture]
    weights_filename = dict(
            alexnet='alexnet_places365-92864cf6.pth',
            vgg16='vgg16_places365-0bafbc55.pth',
            resnet152='resnet152_places365-f928166e5c.pth')[architecture]
    model = model_factory(num_classes=365)
    baseurl = 'https://dissect.csail.mit.edu/models/'
    url = baseurl + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model.load_state_dict(sd)
    model.eval()
    return model

def load_dataset(domain, split=None, full=False, crop_size=None, download=True):
    if domain in ['places', 'imagenet']:
        if split is None:
            split = 'val'
        dirname = 'datasets/%s/%s' % (domain, split)
        if download and not os.path.exists(dirname) and domain == 'places':
            os.makedirs('datasets', exist_ok=True)
            torchvision.datasets.utils.download_and_extract_archive(
                'https://dissect.csail.mit.edu/datasets/' +
                'places_%s.zip' % split,
                'datasets',
                md5=dict(val='593bbc21590cf7c396faac2e600cd30c',
                         train='d1db6ad3fc1d69b94da325ac08886a01')[split])
        places_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(crop_size or 224),
            torchvision.transforms.ToTensor(),
            renormalize.NORMALIZER['imagenet']])
        return parallelfolder.ParallelImageFolders([dirname],
                classification=True,
                shuffle=True,
                transform=places_transform)

def load_segmenter(segmenter_name='netpqc'):
    '''Loads the segementer.'''
    all_parts = ('p' in segmenter_name)
    quad_seg = ('q' in segmenter_name)
    textures = ('x' in segmenter_name)
    colors = ('c' in segmenter_name)

    segmodels = []
    segmodels.append(segmenter.UnifiedParsingSegmenter(segsizes=[256],
            all_parts=all_parts,
            segdiv=('quad' if quad_seg else None)))
    if textures:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'texture')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="texture", segarch=("resnet18dilated", "ppm_deepsup")))
    if colors:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'color')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="color", segarch=("resnet18dilated", "ppm_deepsup")))
    if len(segmodels) == 1:
        segmodel = segmodels[0]
    else:
        segmodel = segmenter.MergedSegmenter(segmodels)
    seglabels = [l for l, c in segmodel.get_label_and_category_names()[0]]
    segcatlabels = segmodel.get_label_and_category_names()[0]
    return segmodel, seglabels, segcatlabels

if __name__ == '__main__':
    main()

