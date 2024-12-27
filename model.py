class Autoencoder(tf.keras.Model):
    def __init__(self, n_emb, channels, n_res, com_cost=0.25):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(channels, n_res)
        self.conv1 = layers.Conv2D(channels[-1], kernel_size=3, strides=1, padding="same")
        self.vector_quantizer = VectorQuantizer(channels[-1], n_emb, com_cost)        
        self.conv2 = layers.Conv2D(channels[-1], kernel_size=3, strides=1, padding="same")
        self.decoder = Decoder(channels[::-1], n_res)
        self.tanh = layers.Activation("tanh")

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        x = self.conv1(x)
        x, quantization_loss = self.vector_quantizer(x)
        x = self.conv2(x)
        x = self.decoder(x, training=training)
        x = self.tanh(x)

        return x, quantization_loss
    
class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = layers.Conv2D(hidden_channels, kernel_size=3, strides=1, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(output_channels, kernel_size=3, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        # Add a projection layer if input and output channels differ
        if input_channels != output_channels:
            self.projection = layers.Conv2D(output_channels, kernel_size=1, strides=1, padding="same")
        else:
            self.projection = None

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Use projection if needed
        if self.projection:
            inputs = self.projection(inputs)

        return self.relu(inputs + x)  # Adding the input (residual connection)
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels):
        super(EncoderBlock, self).__init__()
        self.net = tf.keras.Sequential([
            layers.Conv2D(output_channels, kernel_size=4, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, channels, n_res):
        super(Encoder, self).__init__()
        self.net = tf.keras.Sequential()
        for i in range(len(channels) - 1):
            self.net.add(EncoderBlock(channels[i], channels[i + 1]))
        self.net.add(layers.Conv2D(channels[-1], kernel_size=3, strides=1, padding="same"))
        for i in range(n_res):
            self.net.add(ResidualLayer(channels[-1], channels[-1], channels[-1]))

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

class VectorQuantizer(layers.Layer):
    def __init__(self, num_emb, dim_emb, com_cost):
        super(VectorQuantizer, self).__init__()
        self.num_emb = num_emb  # Number of embeddings
        self.dim_emb = dim_emb  # Dimensionality of each embedding
        self.com_cost = com_cost  # Commitment cost
        
        # Initialize the embeddings
        initializer = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        self.embedding = self.add_weight(
            shape=(num_emb, dim_emb),
            initializer=initializer,
            trainable=True,
            name="embedding"
        )

    def call(self, inputs):
        # Convert input from BCHW to BHWC
        inputs = tf.transpose(inputs, [0, 2, 3, 1])  # BCHW -> BHWC
        input_shape = tf.shape(inputs)

        # Flatten the input for distance calculation
        flat_inputs = tf.reshape(inputs, [-1, self.dim_emb])

        # Calculate squared Euclidean distances
        distances = (
            tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True) + 
            tf.reduce_sum(self.embedding**2, axis=1) - 
            2 * tf.matmul(flat_inputs, self.embedding, transpose_b=True)
        )

        # Encoding: Find the closest embedding index for each input
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_emb)

        # Quantize: Map the indices to the embedding vectors
        quantized = tf.matmul(encodings, self.embedding)
        quantized = tf.reshape(quantized, input_shape)

        # Compute the loss
        e_latent_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized) - inputs))  # Commitment loss
        q_latent_loss = tf.reduce_mean(tf.square(quantized - tf.stop_gradient(inputs)))  # Quantization loss
        loss = q_latent_loss + self.com_cost * e_latent_loss

        # Ensure gradients flow correctly
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        # Convert quantized back to BCHW
        quantized = tf.transpose(quantized, [0, 3, 1, 2])  # BHWC -> BCHW

        return quantized, loss


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels):
        super(DecoderBlock, self).__init__()
        self.net = tf.keras.Sequential([
            layers.Conv2DTranspose(output_channels, kernel_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()  
        ])

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, channels, n_res):
        super(Decoder, self).__init__()
        self.layers = []

        for c in range(len(channels) - 1):
            # Add residual layers at the second block (c == 1)
            if c == 1:
                for i in range(n_res):
                    self.layers.append(ResidualLayer(channels[c], channels[c], channels[c]))

            # Add DecoderBlock layers
            self.layers.append(DecoderBlock(channels[c], channels[c + 1]))

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        return x

