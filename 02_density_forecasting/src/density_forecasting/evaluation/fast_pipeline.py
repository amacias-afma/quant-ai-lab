import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
sys.path.append('..')
from density_forecasting.data.data_loader import fetch_asset_data, fetch_macro_features
from density_forecasting.features.features import create_features
from density_forecasting.models.tf_neural_networks import LinearStudentTNet, WideAndDeepStudentTNet, ConvWideAndDeepNet, nll_loss_fn, prior_guided_loss


def read_data_features(ticker, start_date, end_date, window_size_long=22*6):
    print(f"Fetching {ticker} data...")
    df_asset = fetch_asset_data(ticker=ticker, start=start_date, end=end_date, features=['volume'])

    if ticker == 'USDCLP=X':
        print('entra en ticker USDCLP')
        print(df_asset[df_asset['prices'] < 400])
        df_asset['prices_t1'] = df_asset['prices'].shift()
        df_asset.loc[df_asset['prices'] < 200, 'prices'] = df_asset.loc[df_asset['prices'] < 200, 'prices_t1']
        df_asset['returns'] = np.log(df_asset['prices']).diff().dropna()

    print("Fetching VIX data...")
    df_vix_raw = fetch_asset_data(ticker='^VIX', start=start_date, end=end_date, features=[])
    df_vix = (df_vix_raw[['prices']] / np.sqrt(252)) * 100
    df_vix.columns = ['VIX']

    df_macro_features = fetch_macro_features(start_date, end_date)
    df_asset = pd.concat([df_asset, df_macro_features], axis=1)

    df_features = create_features(df_asset, df_vix, window_size_long, n_lags=3)
    # df_features['vix_ratio'] = df_features['VIX'] / df_features['VIX'].rolling(window_size_long).mean()

    # df_features['ret_1_nu'] = 4 + (6 / df_features['ret_1_kurt'])
    # df_features.loc[df_features['ret_1_nu'] > 30, 'ret_1_nu'] = 30
    # df_features.loc[df_features['ret_1_nu'] < 2, 'ret_1_nu'] = 2

    # window_size_short = int(22 / 1)
    # df_features_st = create_features(df_asset, df_vix, window_size_short, n_lags=1)
    # df_features_st.dropna(inplace=True)
    # df_features_st = df_features_st.astype('float32')
    # df_features_st = df_features_st[['ret_1_mean', 'ret_1_std', 'ret_1_kurt', 'ret_1_skew']]
    # df_features_st.columns = ['ret_1_mean_st', 'ret_1_std_st', 'ret_1_kurt_st', 'ret_1_skew_st']

    # df_features = pd.concat([df_features, df_features_st], axis=1)
    # delta = df_features['prices'].diff()
    # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # df_features['Feat_RSI_14'] = 100 - (100 / (1 + (gain / loss)))

    df_features.dropna(inplace=True)
    df_features = df_features.astype('float32')
    return df_features

def extract_array(df_features, df_y):
    loc = df_features['returns_m1_mean'].values
    scale = df_features['returns_m1_std'].values
    df = df_features['returns_m1_nu'].values
    vix_ratio = df_features['vix_ratio'].values
    ret = df_y.values
    data = {'loc': loc, 'scale': scale, 'df': df, 'vix_ratio': vix_ratio, 'ret': ret}
    return data

def extract_df_nn(df_features, columns_linear, columns_deep):
    df_X_linear = df_features[columns_linear]
    df_X_deep = df_features[columns_deep] if len(columns_deep) > 0 else None
    df_y = df_features['returns']
    return df_X_linear, df_X_deep, df_y

def run_baseline(data_train, last_optimal_beta, cold_start):
    vix_ratio_train = data_train['vix_ratio']
    scale_train = data_train['scale']
    loc_train = data_train['loc']
    ret_train = data_train['ret']
    df_train = data_train['df']

    def objective_function(beta):
        b = beta[0]
        if b < -0.5 or b > 5.0:
            return 1e9

        vix_factor = vix_ratio_train ** b
        vix_factor = vix_factor / np.mean(vix_factor)
        s_adj = np.clip(scale_train * vix_factor, 1e-6, 100.0)
        nll   = -np.sum(t.logpdf(ret_train, df=df_train, loc=loc_train, scale=s_adj))
        return 1e9 if (np.isnan(nll) or np.isinf(nll)) else nll

    start_guess = last_optimal_beta if last_optimal_beta > 0.05 else 0.5
    maxiter     = 200 if cold_start else 30

    try:
        res = minimize(
            objective_function,
            x0=[start_guess],
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'disp': False},
        )
        optimal_beta = np.clip(res.x[0], -0.5, 5.0)
    except Exception:
        optimal_beta = last_optimal_beta

    return optimal_beta

def backtesting_baseline(data_test, optimal_beta):
    vix_ratio_test = data_test['vix_ratio']
    scale_test = data_test['scale']
    loc_test = data_test['loc']
    df_test = data_test['df']
    ret_test = data_test['ret']

    pred_factor = (vix_ratio_test ** optimal_beta).astype('float32')
    historical_factor_mean = np.mean(pred_factor)
    pred_factor_normalized = pred_factor / historical_factor_mean

    scale_pred = np.clip(scale_test * pred_factor_normalized, 1e-6, 100.0).astype('float32')

    pred_mu_flat = tf.reshape(loc_test, [-1])
    pred_sigma_flat = tf.reshape(scale_pred, [-1])
    pred_nu_flat = tf.reshape(df_test, [-1])
    y_test_flat = tf.reshape(ret_test, [-1])
    
    test_dist = tfd.StudentT(df=pred_nu_flat, loc=pred_mu_flat, scale=pred_sigma_flat)

    return {
        'pit_values': test_dist.cdf(y_test_flat).numpy(),
        'mu': pred_mu_flat,
        'sigma': pred_sigma_flat,
        'nu': pred_nu_flat
    }

def initializate_nn_model(columns_linear, model_class, lr, columns_deep=[], lambda_reg=0.1):
    if model_class == 'Linear':
        model = LinearStudentTNet()
    elif model_class == 'WideAndDeep':
        model = WideAndDeepStudentTNet()
    else:
        model = ConvWideAndDeepNet()
        model_class = 'ConvWideAndDeepNet'
        # raise ValueError("model_class must be 'Linear' or 'WideAndDeep'")
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    num_linear_features = len(columns_linear) 
    dummy_x_linear = tf.zeros((1, num_linear_features))
    num_deep_features = len(columns_deep)

    if num_deep_features == 0:
        model(dummy_x_linear)
        input_signature = [
            tf.TensorSpec(shape=[None, num_linear_features], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32)
        ]
    else:
        # --> CRITICAL CNN FIX HERE <--
        if model_class == 'ConvWideAndDeepNet':
            lookback = 20
            # CNN needs 3D data: (batch_size, lookback_window, features)
            dummy_x_deep = tf.zeros((1, lookback, num_deep_features))
        else:
            # Standard Dense network needs 2D data: (batch_size, features)
            dummy_x_deep = tf.zeros((1, num_deep_features))
            
        model((dummy_x_linear, dummy_x_deep))

        # dummy_x_deep = tf.zeros((1, num_deep_features))
        # model((dummy_x_linear, dummy_x_deep))
        # input_signature = [
        #     tf.TensorSpec(shape=[None, num_linear_features], dtype=tf.float32),
        #     tf.TensorSpec(shape=[None], dtype=tf.float32),
        #     tf.TensorSpec(shape=[None, num_deep_features], dtype=tf.float32)
        # ]

    @tf.function
    def train_step(X_lin_batch, y_batch, prior_data=None, X_deep_batch=None):
        with tf.GradientTape() as tape:
            if X_deep_batch is not None:
                mu, sigma, nu = model((X_lin_batch, X_deep_batch))
            else:
                mu, sigma, nu = model(X_lin_batch)
            total_loss, _ = prior_guided_loss(mu, sigma, nu, y_batch, prior_data, lambda_reg=lambda_reg)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss

    @tf.function
    def val_step(X_lin_batch, y_batch, prior_data=None, X_deep_batch=None):
        if X_deep_batch is not None:
            mu, sigma, nu = model((X_lin_batch, X_deep_batch))
        else:
            mu, sigma, nu = model(X_lin_batch)
        _, nll_loss = prior_guided_loss(mu, sigma, nu, y_batch, prior_data, lambda_reg=lambda_reg)
        return nll_loss

    return train_step, val_step, model

def backtesting_neural_networks(X_linear_test, X_deep_test, y_test, model):
    if X_deep_test is not None:
        pred_mu_test, pred_sigma_test, pred_nu_test = model((X_linear_test, X_deep_test))
    else:
        pred_mu_test, pred_sigma_test, pred_nu_test = model(X_linear_test)
    
    pred_mu_flat = tf.reshape(pred_mu_test, [-1])
    pred_sigma_flat = tf.reshape(pred_sigma_test, [-1])
    pred_nu_flat = tf.reshape(pred_nu_test, [-1])
    y_test_flat = tf.reshape(y_test, [-1])
    
    test_dist = tfd.StudentT(df=pred_nu_flat, loc=pred_mu_flat, scale=pred_sigma_flat)
    
    return {
        'pit_values': test_dist.cdf(y_test_flat).numpy(),
        'mu': pred_mu_flat,
        'sigma': pred_sigma_flat,
        'nu': pred_nu_flat
    }

def run_neural_networks(
    df_X_linear_train, df_X_deep_train, df_y_train, 
    df_X_linear_val, df_X_deep_val, df_y_val, 
    current_epochs, train_step, val_step, model, 
    prior_data_train,
    prior_data_v, 
    verbose=False
):
    X_linear_train = tf.convert_to_tensor(df_X_linear_train, dtype=tf.float32)
    X_deep_train = tf.convert_to_tensor(df_X_deep_train, dtype=tf.float32) if df_X_deep_train is not None else None
    y_train = tf.convert_to_tensor(df_y_train, dtype=tf.float32)

    X_linear_val = tf.convert_to_tensor(df_X_linear_val, dtype=tf.float32)
    X_deep_val = tf.convert_to_tensor(df_X_deep_val, dtype=tf.float32) if df_X_deep_val is not None else None
    y_val = tf.convert_to_tensor(df_y_val, dtype=tf.float32)
    
    # X_linear_train = tf.convert_to_tensor(df_X_linear_train.to_numpy(), dtype=tf.float32)
    # X_deep_train = tf.convert_to_tensor(df_X_deep_train.to_numpy(), dtype=tf.float32) if df_X_deep_train is not None else None
    # y_train = tf.convert_to_tensor(df_y_train.to_numpy(), dtype=tf.float32)

    # X_linear_val = tf.convert_to_tensor(df_X_linear_val.to_numpy(), dtype=tf.float32)
    # X_deep_val = tf.convert_to_tensor(df_X_deep_val.to_numpy(), dtype=tf.float32) if df_X_deep_val is not None else None
    # y_val = tf.convert_to_tensor(df_y_val.to_numpy(), dtype=tf.float32)


    if prior_data_train is not None:
        prior_data_train_tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in prior_data_train.items()}
    else:
        prior_data_train_tensors = None
    if prior_data_v is not None:
        prior_data_v_tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in prior_data_v.items()}
    else:
        prior_data_v = None
    # print(prior_data_train_tensors)
    # print(f'prior_data_train_tensors {prior_data_train_tensors}')
    # print(f'prior_data_v_tensors {prior_data_v_tensors}')
    best_val_loss = np.inf
    best_weights = model.get_weights()
    patience_counter = 0
    patience = 20
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(current_epochs):
        
        train_loss = train_step(X_linear_train, y_train, prior_data_train_tensors, X_deep_train).numpy()
        val_loss = val_step(X_linear_val, y_val, prior_data_v_tensors, X_deep_val).numpy()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if val_loss < best_val_loss * 0.999:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}. Restoring best weights.")
            break

        if verbose and (epoch % 50 == 0 or epoch == current_epochs - 1):
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    # Restore the best weights
    model.set_weights(best_weights)
    if verbose:
        print(f'Training finished. Best Val Loss: {best_val_loss:.4f}')
        
    return train_loss_history, val_loss_history
