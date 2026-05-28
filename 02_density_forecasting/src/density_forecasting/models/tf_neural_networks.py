import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class LinearStudentTNet(tf.keras.Model):
    """
    Linear baseline for Student-t density forecasting.
    """
    def __init__(self, init_nu_bias=7.9, init_sigma_bias=0.02):
        super(LinearStudentTNet, self).__init__()
        
        # 1. Initialize Mu to start exactly at a flat 0.0
        self.mu_head = tf.keras.layers.Dense(
            1, 
            kernel_initializer='zeros', 
            bias_initializer='zeros'
        )
        
        # 2. Initialize Sigma 
        self.sigma_head = tf.keras.layers.Dense(
            1, 
            kernel_initializer='zeros', 
            bias_initializer=tf.keras.initializers.Constant(init_sigma_bias)
        )
        
        # 3. Initialize Nu
        self.nu_head = tf.keras.layers.Dense(
            1, 
            activation='softplus',
            kernel_initializer='zeros', 
            bias_initializer=tf.keras.initializers.Constant(init_nu_bias)
        )

    def call(self, inputs):
        mu = self.mu_head(inputs)
        raw_sigma = self.sigma_head(inputs)
        sigma = tf.nn.relu(raw_sigma) + 1e-6
        
        nu = self.nu_head(inputs) + 2.1

        return mu, sigma, nu


class WideAndDeepStudentTNet(tf.keras.Model):
    def __init__(self, prior_mu=0.0005, prior_sigma=0.018, prior_nu=4.5):
        super(WideAndDeepStudentTNet, self).__init__()
        
        # --- THE DEEP PATH (Starts completely asleep) ---
        self.hidden = tf.keras.layers.Dense(16, activation='swish', 
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = tf.keras.layers.Dropout(0.5)

        self.mu_curve = tf.keras.layers.Dense(1, kernel_initializer='zeros')
        self.sigma_curve = tf.keras.layers.Dense(1, kernel_initializer='zeros')
        self.nu_curve = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer='zeros')

        # --- THE WIDE PATH (Linear) ---
        self.mu_linear = tf.keras.layers.Dense(
            1, 
            kernel_initializer='zeros', 
            bias_initializer=tf.keras.initializers.Constant(prior_mu)
        )
        # self.mu_linear = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')
        self.sigma_linear = tf.keras.layers.Dense(
            1, 
            kernel_initializer='zeros', 
            bias_initializer=tf.keras.initializers.Constant(prior_sigma)
        )
        # self.sigma_linear = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant(0.02))
        self.nu_linear = tf.keras.layers.Dense(
            1, 
            activation='softplus', 
            kernel_initializer='zeros', 
            bias_initializer=tf.keras.initializers.Constant(prior_nu - 2.1)
        )
        # self.nu_linear = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant(2.9))

    def call(self, inputs):
        # 1. UNPACK THE INPUTS!
        # When we call the model, we will pass a tuple: (X_linear, X_deep)
        X_linear, X_deep = inputs
        
        # 2. ROUTE THE DEEP FEATURES
        # Only the deep features (e.g., RSI, VIX, Volume) go into the hidden layer
        h = self.hidden(X_deep)
        
        # 3. ROUTE THE LINEAR FEATURES AND ADD THEM TOGETHER
        # Only the linear features (e.g., Moving Averages, Realized Vol) go to the linear layers
        raw_mu = self.mu_linear(X_linear) + self.mu_curve(h)
        raw_sigma = self.sigma_linear(X_linear) + self.sigma_curve(h)
        raw_nu = self.nu_linear(X_linear) + self.nu_curve(h)
        
        # 4. Apply safety limits
        sigma = tf.nn.relu(raw_sigma) + 1e-6
        nu = raw_nu + 2.1

        # nu = tf.nn.relu(raw_nu) + 2.1
        
        return raw_mu, sigma, nu
    
# class WideAndDeepStudentTNet(tf.keras.Model):
#     """
#     Wide & Deep network combining linear baselines with non-linear curve modifiers.
#     """
#     def __init__(self, init_nu_bias=10.9, init_sigma_bias=0.02):
#         super(WideAndDeepStudentTNet, self).__init__()
        
#         # --- THE DEEP PATH (Non-Linear) ---
#         # We use 'swish' (x * sigmoid(x)) because it draws perfectly smooth polynomials
#         self.hidden = tf.keras.layers.Dense(16, activation='swish')
        
#         # --- THE WIDE PATH (Strictly Linear baselines) ---
#         self.mu_linear = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')
#         self.sigma_linear = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant(init_sigma_bias))
#         self.nu_linear = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant(init_nu_bias))
        
#         # --- THE CURVE MODIFIERS (Outputs of the hidden layer) ---
#         # These start at zero, meaning on Epoch 1, the network is PURELY LINEAR
#         self.mu_curve = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')
#         self.sigma_curve = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')
#         self.nu_curve = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer='zeros', bias_initializer='zeros')

#     def call(self, inputs):
#         # 1. Calculate the curves
#         h = self.hidden(inputs)
        
#         # 2. Add the Strict Linear Baseline to the Smooth Non-Linear Curves!
#         raw_mu = self.mu_linear(inputs) + self.mu_curve(h)
#         raw_sigma = self.sigma_linear(inputs) + self.sigma_curve(h)
#         raw_nu = self.nu_linear(inputs) + self.nu_curve(h)

#         # 3. Apply safety limits
#         sigma = tf.nn.relu(raw_sigma) + 1e-6
#         nu = raw_nu + 2.1
        
#         return raw_mu, sigma, nu

def nll_loss_fn(mu_pred, sigma_pred, nu_pred, y_target):
    """
    Negative Log-Likelihood Loss for Student-t Distribution.
    """
    y_target = tf.reshape(y_target, [-1, 1])
    dist = tfd.StudentT(df=nu_pred, loc=mu_pred, scale=sigma_pred)
    return -tf.reduce_mean(dist.log_prob(y_target))

# ==========================================
# 2. THE PRIOR-GUIDED NLL LOSS FUNCTION
# ==========================================
def prior_guided_loss(mu_pred, sigma_pred, nu_pred, y_target, prior_data=None, lambda_reg=0.05):
    # A. Standard Negative Log-Likelihood (The data-driven force)
    dist = tfd.StudentT(df=nu_pred, loc=mu_pred, scale=sigma_pred)
    nll_loss = -tf.reduce_mean(dist.log_prob(y_target))
    
    # B. The Prior Penalties (The regularizing force pushing back to the baseline)
    # We use Mean Squared Error between the network's predictions and the parametric prior
    if prior_data is not None:
        mu_penalty = tf.reduce_mean(tf.square(tf.reshape(mu_pred, [-1]) - tf.reshape(prior_data['mu'], [-1])))
        sigma_penalty = tf.reduce_mean(tf.square(tf.reshape(sigma_pred, [-1]) - tf.reshape(prior_data['sigma'], [-1])))
        nu_penalty = tf.reduce_mean(tf.square(tf.reshape(nu_pred, [-1]) - tf.reshape(prior_data['nu'], [-1])))
        # Total combined loss
        total_loss = nll_loss + lambda_reg * (mu_penalty + sigma_penalty + nu_penalty)
    else:
        total_loss = nll_loss
    return total_loss

def train_expanding_window_model(df_X_linear, df_y, df_X_deep=None, model_class='Linear', epochs=500, test_window=22, porcentage_train=0.6, lr=0.015):
    """
    Trains a model using an expanding window approach and collects predictions out-of-sample.
    """
    n_train = int(porcentage_train * len(df_X_linear))
    
    pit_values_list = []
    dates_list = []
    
    pred_mu_list = []
    pred_sigma_list = []
    pred_nu_list = []

    print(f"Training {model_class} Network on Polynomial Data...")
    
    
    if model_class == 'Linear':
        model = LinearStudentTNet()
    elif model_class == 'WideAndDeep':
        model = WideAndDeepStudentTNet()
    else:
        raise ValueError("model_class must be 'Linear' or 'WideAndDeep'")
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Initialize model weights by calling it on dummy data
    num_linear_features = df_X_linear.shape[1] 
    dummy_x_linear = tf.zeros((1, num_linear_features))

    num_deep_features = df_X_deep.shape[1] if df_X_deep is not None else 0

    if num_deep_features == 0:
        model(dummy_x_linear)
    else:
        dummy_x_deep = tf.zeros((1, num_deep_features))
        model((dummy_x_linear, dummy_x_deep))

    # initial_weights = model.get_weights()
        
    if num_deep_features == 0:
        input_signature = [
            tf.TensorSpec(shape=[None, num_linear_features], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32) # y is 1D series
        ]
    else:
        input_signature = [
            tf.TensorSpec(shape=[None, num_linear_features], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32), # y is 1D series
            tf.TensorSpec(shape=[None, num_deep_features], dtype=tf.float32)
        ]
    @tf.function(input_signature=input_signature)
    def train_step(X_lin_batch, y_batch, X_deep_batch=None):
        with tf.GradientTape() as tape:
            if X_deep_batch is not None:
                mu, sigma, nu = model((X_lin_batch, X_deep_batch))
            else:
                mu, sigma, nu = model(X_lin_batch)
            loss = nll_loss_fn(mu, sigma, nu, y_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    is_first_window = True

    while n_train < len(df_X_linear):
        current_epochs = epochs if is_first_window else max(1, epochs // 2)
        
        print('-'*30)
        print(f"Train size: {n_train}, Total data: {len(df_X_linear)}, Epochs: {current_epochs} ({'Cold Start' if is_first_window else 'Warm Start'})")
        print('-'*30)
        
        # We no longer reset model.set_weights(initial_weights) or the optimizer variables
        # This gives us a true "warm start" from the previous window's state.
            
        df_X_linear_train = df_X_linear.iloc[:n_train]
        df_X_deep_train = df_X_deep.iloc[:n_train] if df_X_deep is not None else None
        df_y_train = df_y.iloc[:n_train]

        df_X_linear_test = df_X_linear.iloc[n_train:n_train + test_window]
        df_X_deep_test = df_X_deep.iloc[n_train:n_train + test_window] if df_X_deep is not None else None
        df_y_test = df_y.iloc[n_train:n_train + test_window]

        X_linear_train = tf.convert_to_tensor(df_X_linear_train.to_numpy(), dtype=tf.float32)
        X_deep_train = tf.convert_to_tensor(df_X_deep_train.to_numpy(), dtype=tf.float32) if df_X_deep_train is not None else None
        y_train = tf.convert_to_tensor(df_y_train.to_numpy(), dtype=tf.float32)
        X_linear_test = tf.convert_to_tensor(df_X_linear_test.to_numpy(), dtype=tf.float32)
        X_deep_test = tf.convert_to_tensor(df_X_deep_test.to_numpy(), dtype=tf.float32) if df_X_deep_test is not None else None
        y_test = tf.convert_to_tensor(df_y_test.to_numpy(), dtype=tf.float32)
        
        for epoch in range(current_epochs):
            # print(f"Epoch {epoch+1}/{current_epochs} - ", end="")
            # print(X_linear_train.shape, X_deep_train.shape, y_train.shape)
            loss = train_step(X_linear_train, y_train, X_deep_train)
            if epoch % 100 == 0 or epoch == current_epochs - 1:
                print(f"Epoch {epoch:03d} | NLL Loss: {loss.numpy():.4f}")
                
        is_first_window = False

        # Test results
        if X_deep_test is not None:
            pred_mu_test, pred_sigma_test, pred_nu_test = model((X_linear_test, X_deep_test))
        else:
            pred_mu_test, pred_sigma_test, pred_nu_test = model(X_linear_test)

        dates_list.append(df_X_linear_test.index)
        
        pred_mu_flat = tf.reshape(pred_mu_test, [-1])
        pred_sigma_flat = tf.reshape(pred_sigma_test, [-1])
        pred_nu_flat = tf.reshape(pred_nu_test, [-1])
        
        pred_mu_list.append(pred_mu_flat.numpy())
        pred_sigma_list.append(pred_sigma_flat.numpy())
        pred_nu_list.append(pred_nu_flat.numpy())
        
        y_test_flat = tf.reshape(y_test, [-1])
        test_dist = tfd.StudentT(
            df=pred_nu_flat, 
            loc=pred_mu_flat, 
            scale=pred_sigma_flat
        )
        pit_values_list.append(test_dist.cdf(y_test_flat).numpy())
        
        n_train += test_window
        
    dates_test_total = pd.Index(np.concatenate(dates_list)) if dates_list else pd.Index([])
    pred_mu_test_total = np.concatenate(pred_mu_list) if pred_mu_list else np.array([])
    pred_sigma_test_total = np.concatenate(pred_sigma_list) if pred_sigma_list else np.array([])
    pred_nu_test_total = np.concatenate(pred_nu_list) if pred_nu_list else np.array([])
    pit_values_test_total = np.concatenate(pit_values_list) if pit_values_list else np.array([])
    
    return dates_test_total, pred_mu_test_total, pred_sigma_test_total, pred_nu_test_total, pit_values_test_total
