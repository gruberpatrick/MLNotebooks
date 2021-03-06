
from keras import layers, models, optimizers, regularizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    # --------------------------------------------------------------------------
    def __init__(self, state_size, action_size, lr=.0001):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.lr = lr
        self.dropout = .2

        self.build_model()

    # --------------------------------------------------------------------------
    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        ########################################################################
        # Add hidden layer(s) for state pathway
        X = layers.Dense(units=256, kernel_initializer='glorot_normal')(states)
        #X = layers.BatchNormalization()(X)
        X = layers.Activation("relu")(X)
        #X = layers.Dropout(self.dropout)(X)

        X = layers.Dense(units=256, kernel_initializer='glorot_normal')(X)
        #X = layers.BatchNormalization()(X)
        net_states = layers.Activation("relu")(X)

        # Add hidden layer(s) for action pathway
        X = layers.Dense(units=256, kernel_initializer='glorot_normal')(actions)
        #X = layers.BatchNormalization()(X)
        X = layers.Activation("relu")(X)
        #X = layers.Dropout(self.dropout)(X)

        X = layers.Dense(units=256, kernel_initializer='glorot_normal')(X)
        #X = layers.BatchNormalization()(X)
        net_actions = layers.Activation("relu")(X)
        ########################################################################

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values', kernel_initializer='glorot_normal')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
