import imp

def load_positional_embedder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).get_embedder

def load_canonical_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).CanonicalMLP

def load_mweight_vol_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).MotionWeightVolumeDecoder

def load_pose_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).BodyPoseRefiner

def load_non_rigid_motion_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).NonRigidMotionMLP

def load_transformation(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).Transformation

def load_encoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).Encoder

def load_fuse_net(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).FuseNetwork

def load_sparse_conv_net(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).SparseConvNet

def load_discriminator(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).Discriminator
