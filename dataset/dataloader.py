from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())
    if (args.dataset == 'CUFED'):
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}
    
    elif (args.dataset == 'IMM'):
        data_train = getattr(m, 'TrainSet')(args) #get Returnvalue of IMM.py
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}
        
    elif (args.dataset == 'IMMRW'):
        dataloader_train = {}
        if not args.eval:
            data_train = getattr(m, 'TrainSet')(args) #get Returnvalue of IMM.py
            dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(1):#vorher 5
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        data_ref = getattr(m, 'RefSet')(args)
        dataloader = {'train': dataloader_train, 'test': dataloader_test, 'ref': data_ref}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader