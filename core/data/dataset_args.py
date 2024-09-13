from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394', '396']

    start_frame_train = {'313': 0, '315': 0, '377': 0, '386': 0, '387': 0, '390': 600, '392': 0, '393': 0, '394': 0, '396': 0 }
    end_frame_train = {'313': 550, '315': 550, '377': 550, '386': 550, '387': 550, '390': 1150, '392': 550, '393': 550, '394': 550, '396': 550 }
    start_frame_test = {'313': 0, '315': 0, '377': 0, '386': 0, '387': 0, '390': 600, '392': 0, '393': 0, '394': 0, '396': 0 }
    end_frame_test = {'313': 550, '315': 550, '377': 550, '386': 550, '387': 550, '390': 1150, '392': 550, '393': 550, '394': 550, '396': 550 }

    start_frame_train_100frame = {'313': 0, '315': 0, '377': 0, '386': 0, '387': 0, '390': 600, '392': 0, '393': 0, '394': 0, '396': 0 }
    end_frame_train_100frame = {'313': 101, '315': 101, '377': 101, '386': 101, '387': 101, '390': 701, '392': 101, '393': 101, '394': 101, '396': 101 }
    start_frame_test_100frame = {'313': 301, '315': 301, '377': 301, '386': 301, '387': 301, '390': 301, '392': 301, '393': 301, '394': 301, '396': 301 }
    end_frame_test_100frame = {'313': 550, '315': 550, '377': 550, '386': 550, '387': 550, '390': 1150, '392': 550, '393': 550, '394': 550, '396': 550 }

    start_frame_train_300frame = {'313': 0, '315': 0, '377': 0, '386': 0, '387': 0, '390': 600, '392': 0, '393': 0, '394': 0, '396': 0 }
    end_frame_train_300frame = {'313': 301, '315': 301, '377': 301, '386': 301, '387': 301, '390': 901, '392': 301, '393': 301, '394': 301, '396': 301 }
    start_frame_test_300frame = {'313': 301, '315': 301, '377': 301, '386': 301, '387': 301, '390': 901, '392': 301, '393': 301, '394': 301, '396': 301 }
    end_frame_test_300frame = {'313': 550, '315': 550, '377': 550, '386': 550, '387': 550, '390': 1150, '392': 550, '393': 550, '394': 550, '396': 550 }

    AIST_subjects = ['d01','d02','d03','d04','d05','d06','d07','d08','d09','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','d21','d22','d23','d24','d25','d26','d27','d28','d29','d30']
    AIST_start_frame_train = {'d01':0, 'd02':0, 'd03':0, 'd04':0,'d05':0, 'd06':0,'d07':0, 'd08':0,'d09':0, 'd10':0,
                              'd11':0, 'd12':0, 'd13':0, 'd14':0,'d15':0, 'd16':0,'d17':0, 'd18':0,'d19':0, 'd20':0,
                              'd21':0, 'd22':0, 'd23':0, 'd24':0,'d25':0, 'd26':0,'d27':0, 'd28':0,'d29':0, 'd30':0}
    AIST_end_frame_train = {'d01':500, 'd02':500, 'd03':500, 'd04':500,'d05':500, 'd06':500,'d07':500, 'd08':500,'d09':500, 'd10':500,
                            'd11':500, 'd12':500, 'd13':500, 'd14':500,'d15':500, 'd16':500,'d17':500, 'd18':500,'d19':500, 'd20':500,
                            'd21':500, 'd22':500, 'd23':500, 'd24':500,'d25':500, 'd26':500,'d27':500, 'd28':500,'d29':500, 'd30':500}
    AIST_start_frame_test = {'d01':0, 'd02':0, 'd03':0, 'd04':0,'d05':0, 'd06':0,'d07':0, 'd08':0,'d09':0, 'd10':0,
                             'd11':0, 'd12':0, 'd13':0, 'd14':0,'d15':0, 'd16':0,'d17':0, 'd18':0,'d19':0, 'd20':0,
                             'd21':0, 'd22':0, 'd23':0, 'd24':0,'d25':0, 'd26':0,'d27':0, 'd28':0,'d29':0, 'd30':0}
    AIST_end_frame_test = {'d01':500, 'd02':500, 'd03':500, 'd04':500,'d05':500, 'd06':500,'d07':500, 'd08':500,'d09':500, 'd10':500,
                           'd11':500, 'd12':500, 'd13':500, 'd14':500,'d15':500, 'd16':500,'d17':500, 'd18':500,'d19':500, 'd20':500,
                           'd21':500, 'd22':500, 'd23':500, 'd24':500,'d25':500, 'd26':500,'d27':500, 'd28':500,'d29':500, 'd30':500}

    AIST_start_frame_train_100frame = {'d01':0, 'd02':0, 'd03':0, 'd04':0,'d05':0, 'd06':0,'d07':0, 'd08':0,'d09':0, 'd10':0,
                              'd11':0, 'd12':0, 'd13':0, 'd14':0,'d15':0, 'd16':0,'d17':0, 'd18':0,'d19':0, 'd20':0,
                              'd21':0, 'd22':0, 'd23':0, 'd24':0,'d25':0, 'd26':0,'d27':0, 'd28':0,'d29':0, 'd30':0}
    AIST_end_frame_train_100frame = {'d01':100, 'd02':100, 'd03':100, 'd04':100,'d05':100, 'd06':100,'d07':100, 'd08':100,'d09':100, 'd10':100,
                            'd11':100, 'd12':100, 'd13':100, 'd14':100,'d15':100, 'd16':100,'d17':100, 'd18':100,'d19':100, 'd20':100,
                            'd21':100, 'd22':100, 'd23':100, 'd24':100,'d25':100, 'd26':100,'d27':100, 'd28':100,'d29':100, 'd30':100}
    AIST_start_frame_test_100frame = {'d01':300, 'd02':300, 'd03':300, 'd04':300,'d05':300, 'd06':300,'d07':300, 'd08':300,'d09':300, 'd10':300,
                            'd11':300, 'd12':300, 'd13':300, 'd14':300,'d15':300, 'd16':300,'d17':300, 'd18':300,'d19':300, 'd20':300,
                            'd21':300, 'd22':300, 'd23':300, 'd24':300,'d25':300, 'd26':300,'d27':300, 'd28':300,'d29':300, 'd30':300}
    AIST_end_frame_test_100frame = {'d01':500, 'd02':500, 'd03':500, 'd04':500,'d05':500, 'd06':500,'d07':500, 'd08':500,'d09':500, 'd10':500,
                           'd11':500, 'd12':500, 'd13':500, 'd14':500,'d15':500, 'd16':500,'d17':500, 'd18':500,'d19':500, 'd20':500,
                           'd21':500, 'd22':500, 'd23':500, 'd24':500,'d25':500, 'd26':500,'d27':500, 'd28':500,'d29':500, 'd30':500}

    AIST_start_frame_train_300frame = {'d01':0, 'd02':0, 'd03':0, 'd04':0,'d05':0, 'd06':0,'d07':0, 'd08':0,'d09':0, 'd10':0,
                              'd11':0, 'd12':0, 'd13':0, 'd14':0,'d15':0, 'd16':0,'d17':0, 'd18':0,'d19':0, 'd20':0,
                              'd21':0, 'd22':0, 'd23':0, 'd24':0,'d25':0, 'd26':0,'d27':0, 'd28':0,'d29':0, 'd30':0}
    AIST_end_frame_train_300frame = {'d01':300, 'd02':300, 'd03':300, 'd04':300,'d05':300, 'd06':300,'d07':300, 'd08':300,'d09':300, 'd10':300,
                            'd11':300, 'd12':300, 'd13':300, 'd14':300,'d15':300, 'd16':300,'d17':300, 'd18':300,'d19':300, 'd20':300,
                            'd21':300, 'd22':300, 'd23':300, 'd24':300,'d25':300, 'd26':300,'d27':300, 'd28':300,'d29':300, 'd30':300}
    AIST_start_frame_test_300frame = {'d01':300, 'd02':300, 'd03':300, 'd04':300,'d05':300, 'd06':300,'d07':300, 'd08':300,'d09':300, 'd10':300,
                            'd11':300, 'd12':300, 'd13':300, 'd14':300,'d15':300, 'd16':300,'d17':300, 'd18':300,'d19':300, 'd20':300,
                            'd21':300, 'd22':300, 'd23':300, 'd24':300,'d25':300, 'd26':300,'d27':300, 'd28':300,'d29':300, 'd30':300}
    AIST_end_frame_test_300frame = {'d01':500, 'd02':500, 'd03':500, 'd04':500,'d05':500, 'd06':500,'d07':500, 'd08':500,'d09':500, 'd10':500,
                           'd11':500, 'd12':500, 'd13':500, 'd14':500,'d15':500, 'd16':-1,'d17':-1, 'd18':-1,'d19':-1, 'd20':-1,
                           'd21':500, 'd22':500, 'd23':500, 'd24':500,'d25':500, 'd26':500,'d27':500, 'd28':500,'d29':500, 'd30':500}



    if cfg.category == 'actorsnerf' and cfg.task == 'zju_mocap':
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"datasets/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "src_type": 'zju_mocap',
                    "frame_start": start_frame_train[sub],
                    "frame_end": end_frame_train[sub],
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"datasets/zju_mocap/{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "frame_start": start_frame_test[sub],
                    "frame_end": end_frame_test[sub],
                },
                f"zju_{sub}_train_100frame": {
                    "dataset_path": f"datasets/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "src_type": 'zju_mocap',
                    "frame_start": start_frame_train_100frame[sub],
                    "frame_end": end_frame_train_100frame[sub],
                },
                f"zju_{sub}_test_100frame": {
                    "dataset_path": f"datasets/zju_mocap/{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "frame_start": start_frame_test_100frame[sub],
                    "frame_end": end_frame_test_100frame[sub],
                },
                f"zju_{sub}_train_300frame": {
                    "dataset_path": f"datasets/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "src_type": 'zju_mocap',
                    "frame_start": start_frame_train_300frame[sub],
                    "frame_end": end_frame_train_300frame[sub],
                },
                f"zju_{sub}_test_300frame": {
                    "dataset_path": f"datasets/zju_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "frame_start": start_frame_test_300frame[sub],
                    "frame_end": end_frame_test_300frame[sub],
                },
            })



    if cfg.category == 'actorsnerf' and cfg.task == 'AIST_mocap':
        for sub in AIST_subjects:
            dataset_attrs.update({
                f"AIST_{sub}_train": {
                    "dataset_path": f"datasets/AIST_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "src_type": 'AIST_mocap',
                    "frame_start": AIST_start_frame_train[sub],
                    "frame_end": AIST_end_frame_train[sub],
                },
                f"AIST_{sub}_test": {
                    "dataset_path": f"datasets/AIST_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'AIST_mocap',
                    "frame_start": AIST_start_frame_test[sub],
                    "frame_end": AIST_end_frame_test[sub],
                },
                f"AIST_{sub}_train_100frame": {
                    "dataset_path": f"datasets/AIST_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "src_type": 'AIST_mocap',
                    "frame_start": AIST_start_frame_train_100frame[sub],
                    "frame_end": AIST_end_frame_train_100frame[sub],
                },
                f"AIST_{sub}_test_100frame": {
                    "dataset_path": f"datasets/AIST_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'AIST_mocap',
                    "frame_start": AIST_start_frame_test_100frame[sub],
                    "frame_end": AIST_end_frame_test_100frame[sub],
                },
                f"AIST_{sub}_train_300frame": {
                    "dataset_path": f"datasets/AIST_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "src_type": 'AIST_mocap',
                    "frame_start": AIST_start_frame_train_300frame[sub],
                    "frame_end": AIST_end_frame_train_300frame[sub],
                },
                f"AIST_{sub}_test_300frame": {
                    "dataset_path": f"datasets/AIST_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'AIST_mocap',
                    "frame_start": AIST_start_frame_test_300frame[sub],
                    "frame_end": AIST_end_frame_test_300frame[sub],
                },
            })


    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
