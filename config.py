import config

dataSetting={
    'inputSize': 224,
    'classSize': 2,
    'class_to_idx': {'10':0,'30':1}
}

trainSetting={
    'batchSize': 32,
    'lr' : 1e-4,
    'epochs': 1200,
    'modelName': 'model_ConvNext',
}

expSetting={
    'runs': 1
}