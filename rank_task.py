proj_path = "/home.../" # set the absolute path of this project

import sys, os
sys.path.append(proj_path)

import numpy as np
import random
import time
import torch
from torch.utils.data import DataLoader



def get_engine(name, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    print("=========== model: " + name + " ===========") 
    if name == 'final':
        return FinalEngine(Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings)
    else:
        raise ValueError('unknow model name: ' + name)


def setup_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(seed)  # 为所有的GPU设置种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Run(DataSettings, ModelSettings, TrainSettings, ResultSettings,
        mode='train',
        timestamp=None):

    ## =========== setting init ===========
    setup_seed(817)  # random seed

    model_name = ModelSettings['model_name']  # final
    save_dir = ResultSettings['save_dir'] 
    save_dir = save_dir + model_name[0].upper()+model_name[1:] + '/'  
    epoch = eval(TrainSettings['epoch'])  # 500
    batch_size = eval(TrainSettings['batch_size'])  # 4096  128
    s_batch_size = eval(TrainSettings['s_batch_size'])  # 2048  64


    ## =========== data  init ===========
    Sampler = SampleGenerator(DataSettings)
    graphs = Sampler.generateGraphs()

    print('User count: %d. Item count: %d. ' % (Sampler.user_num, Sampler.candidate_num))
    print('Without Negatives, Train count: %d. Validation count: %d. Test count: %d' % (Sampler.train_size, Sampler.val_size, Sampler.test_size))
    print("epioion_64-cat_用userembeding")


    ## =========== model init ===========
    Engine = get_engine(model_name, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings)

    optimizer =Engine.optimizer


    traindata = np.array(Sampler.train_df)
    # traindata 169992个 三元列表[[4306 94324 5],[6608 74911 5] ...]每个列表里 分别代表 用户id 物品 评分 这里np.array作用主要是 由[[4306,94324,5][]..]变成三元列表[[4306 94324 5],[6608 74911 5] ...]
    validdata = np.array(Sampler.val_df)  # validdata 56664三元个列表   [[293 27927 3
    testdata = np.array(Sampler.test_df)  # testdata 56664三元个列表

    train_u = traindata[:, 0]  # x[:,n]就是取所有集合的第n个数据, 只取有关用户id
    # [4306 6608 4214 只取有关用户id
    train_v = traindata[:, 1]
    train_r = traindata[:, 2]

    valid_u = validdata[:, 0]
    valid_v = validdata[:, 1]
    valid_r = validdata[:, 2]

    test_u = testdata[:, 0]
    test_v = testdata[:, 1]
    test_r = testdata[:, 2]

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    # trainset 用户 物品 评分 三个元组 每个元组都是一个169992维的tensor张量
    # train_u  train_v train_r各有169992个
    # (tensor([4306, 6608, 4214,  ...,  799, 6867,   95]), tensor([94324, 74911,  1388,  ..., 50298, 18946, 10247]), tensor([5., 5., 4.,  ..., 5., 4., 3.]))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # 数据封装 DataLoader函数每次抛出一组数据（128）。直至把所有的数据都抛出 总共1329组
    # train_loader：：：  (tensor([4306, 6608, 4214,  ...,  799选128个 tensor([94324, 74911,  1388,  ..., 50298 128个  tensor([5., 5., 4.,  ..., 5 128个 总共1329组
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=s_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=s_batch_size, shuffle=True)

    ## =========== train || inference ===
    if timestamp == None:
        timestamp = time.time()
    localtime = str(time.asctime(time.localtime(int(timestamp))))
    with open(save_dir + model_name + "_" + str(int(timestamp)) + ".txt", "a") as f:
        model_save_dir = save_dir + 'files/' + str(int(timestamp)) + '/'

        f.write('\n\n\n' + '========== ' + localtime + " " + str(int(timestamp)) + ' ==========' + '\n')
        f.write(str(DataSettings) + '\n' + str(ModelSettings) + '\n' + str(TrainSettings) + '\n')
        f.write(str(Engine.model))
        f.write('\n')

        if mode == "train":  # train mode
            print("=========== Training Start ===========")
            # val_hr, val_ndcg = 0, 0
            # test_hr, test_ndcg = 0, 0
            # best_result = ""
            # # endure_count = 0
            # # early_stop_step = 10
            # pre_train_epoch = 0

            best_rmse = 9999.0
            best_mae = 9999.0
            endure_count = 0

            for epoch_i in range(1, epoch + 1):

                ### train
                # Sampler.generateTrainNegative(combine=True)
                # train_loader = DataLoader(PointDataset(Sampler.train_df), batch_size=batch_size, shuffle=True, num_workers=0)
                Engine.train(train_loader, graphs, epoch_i, best_rmse, best_mae)
                # test_pos_loader = DataLoader(RankDataset(Sampler.val_df), batch_size=s_batch_size, shuffle=True,
                #                              num_workers=0)

                expected_rmse, mae = Engine.evaluate(valid_loader, graphs)
                if best_rmse > expected_rmse:
                    best_rmse = expected_rmse
                    best_mae = mae
                    endure_count = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': Engine.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, '/home/...Models/epinion.pt')
                    # f.write('epoch: ' + str(epoch_i) + '\n')
                    # f.write(expected_rmse+ '\n')
                    # torch.save(Engine.model, f'/home.../Models/ciao.pt')
                else:
                    endure_count += 1
                print("rmse on valid set: %.4f, mae:%.4f " % (expected_rmse, mae))
                rmse, mae = Engine.evaluate(test_loader, graphs)
                print('rmse on test set: %.4f, mae:%.4f ' % (rmse, mae))

                if endure_count > 5:
                    break
            print('finished')



