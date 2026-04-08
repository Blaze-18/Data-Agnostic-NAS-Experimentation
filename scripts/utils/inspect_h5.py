import h5py
f='data/processed/dataset.h5'
with h5py.File(f,'r') as hf:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print('D:', name, getattr(obj,'shape',None), getattr(obj,'dtype',None))
        else:
            print('G:', name)
    hf.visititems(walk)
    print('\nTop-level keys:', list(hf.keys()))
