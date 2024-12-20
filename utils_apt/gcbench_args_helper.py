
# type(args): <class 'argparse.Namespace'>

# Consistent with /coreset/train_coreset_induct.py
# Related methods: "random", "herding", "kcenter"
def set_args_coreset(args):
    if args.gpu_id is None:
        args.gpu_id = 0
    if args.hidden is None:
        args.hidden = 256
    if args.normalize_features is None:
        args.normalize_features = True
    if args.keep_ratio is None:
        args.keep_ratio = 1.0
    if args.lr is None:
        args.lr = 0.01
    if args.weight_decay is None:
        args.weight_decay = 5e-4
    if args.dropout is None:
        args.dropout = 0.5
    if args.seed is None:
        args.seed = 15
    if args.nlayers is None:
        args.nlayers = 2
    if args.epochs is None:
        args.epochs = 400
    if args.inductive is None:
        args.inductive = 1
    if args.mlp is None:
        args.mlp = 0
    if args.save is None:
        args.save = 0

# Consistent with DM/main.py
# Related methods: "GCDM"
def set_args_DM(args):
    if args.config is None:
        args.config = "config.json"
    if args.config_dir is None:
        args.config_dir = "configs"
    if args.section is None:
        args.section = ""
    if args.wandb is None:
        args.wandb = 0
    if args.gpu_id is None:
        args.gpu_id = 0
    if args.data_dir is None:
        args.data_dir = "data"
    if args.save_dir is None:
        args.save_dir = "save"
    if args.keep_ratio is None:
        args.keep_ratio = 1.0
    if args.sgc is None:
        args.sgc = 1
    if args.seed is None:
        args.seed = 15
    if args.alpha is None:
        args.alpha = 0
    if args.debug is None:
        args.debug = 0
    if args.save is None:
        args.save = 1
    if args.epochs is None:
        args.epochs = 2000
    if args.nlayers is None:
        args.nlayers = 2
    if args.hidden is None:
        args.hidden = 256
    if args.lr_adj is None:
        args.lr_adj = 1e-3
    if args.lr_feat is None:
        args.lr_feat = 1e-3
    if args.lr_model is None:
        args.lr_model = 1e-2
    if args.weight_decay is None:
        args.weight_decay = 0.0
    if args.dropout is None:
        args.dropout = 0.0
    if args.normalize_features is None:
        args.normalize_features = True
    if args.inner is None:
        args.inner = 0
    if args.outer is None:
        args.outer = 20
    if args.transductive is None:
        args.transductive = 1
    if args.one_step is None:
        args.one_step = 1
    if args.init_way is None:
        args.init_way = "Random_real"
    if args.label_rate is None:
        args.label_rate = 1


# Consistent with GM/main_nc.py
# Related methods: "GCond", "SGDD"
def set_args_GM_NC(args):
    if args.config is None:
        args.config = "config.json"
    if args.config_dir is None:
        args.config_dir = "configs"
    if args.section is None:
        args.section = ""
    if args.wandb is None:
        args.wandb = 0
    if args.wandb_id is None:
        args.wandb_id = ""
    if args.gpu_id is None:
        args.gpu_id = 0
    if args.data_dir is None:
        args.data_dir = "data"
    if args.save_dir is None:
        args.save_dir = "save"
    if args.keep_ratio is None:
        args.keep_ratio = 1.0
    if args.seed is None:
        args.seed = 15
    if args.alpha is None:
        args.alpha = 0
    if args.debug is None:
        args.debug = 0
    if args.save is None:
        args.save = 1
    if args.epochs is None:
        args.epochs = 600
    if args.nlayers is None:
        args.nlayers = 2
    if args.hidden is None:
        args.hidden = 256
    if args.lr_adj is None:
        args.lr_adj = 0.01
    if args.lr_feat is None:
        args.lr_feat = 1e-4
    if args.lr_model is None:
        args.lr_model = 1e-4
    if args.weight_decay is None:
        args.weight_decay = 0.0
    if args.dropout is None:
        args.dropout = 0.0
    if args.normalize_features is None:
        args.normalize_features = True
    if args.sgc is None:
        args.sgc = 1
    if args.gt is None:
        args.gt = 0
    if args.inner is None:
        args.inner = 0
    if args.outer is None:
        args.outer = 20
    if args.transductive is None:
        args.transductive = 1
    if args.one_step is None:
        args.one_step = 0
    if args.option is None:
        args.option = 0
    if args.label_rate is None:
        args.label_rate = 1
    if args.init_way is None:
        args.init_way = "Random"
    if args.dis_metric is None:
        args.dis_metric = "ours"
    if args.beta is None:
        args.beta = 0.5
    if args.early_stopping is None:
        args.early_stopping = 0
    if args.max_epochs_without_improvement is None:
        args.max_epochs_without_improvement = 200
    if args.prune is None:
        args.prune = 0.05
    if args.mining is None:
        args.mining = 0.001
    if args.circulation is None:
        args.circulation = 20
    if args.mx_size is None:
        args.mx_size = 100
    if args.ep_ratio is None:
        args.ep_ratio = 0.5
    if args.sinkhorn_iter is None:
        args.sinkhorn_iter = 10
    if args.opt_scale is None:
        args.opt_scale = 0
    if args.coreset_method is None:
        args.coreset_method = "kcenter"

