import keras 

def activations(type_="relu"):
    match type_:
        case "relu":
            return keras.layers.ReLU()
        case "sigmoid":
            return keras.layers.Lambda(lambda x: keras.activations.sigmoid(x))
        case "tanh":
            return keras.layers.Lambda(lambda x: keras.activations.tanh(x))
        case 'selu':
            return keras.layers.Lambda(lambda x: keras.activations.selu(x))
        case "PReLU":
            return keras.layers.PReLU()
        case "elu":
            return keras.layers.ELU()
        
    return keras.layers.ReLU()

def optimizers(type_="adam", lr=0.0001):
    match type_:
        case "adam":
            return keras.optimizers.Adam(learning_rate=lr)
        case "sgd":
            return keras.optimizers.SGD(learning_rate=lr)

def kernal_initializer(type_="relu"):
    if (type_ in ['relu', 'PReLU', 'elu']):
        return keras.initializers.he_normal()
    elif (type_ in ['selu']):
        return keras.initializers.lecun_normal()
    
    return keras.initializers.glorot_normal()
    
def loss_(type_='mse'):
    match type_:
        case "mse":
            return keras.losses.mean_absolute_error
        case "mae":
            return keras.losses.mean_squared_error
        case "huber":
            return keras.losses.huber
        
def metrics(type_=['mae']):
    def cases_(i):
        match i:
            case "mae":
                return keras.metrics.mean_absolute_error
            case "mse":
                return keras.metrics.mean_squared_error
            case "huber":
                return keras.metrics.huber
        return keras.metrics.mean_absolute_error

    return [cases_(i) for i in type_]
        