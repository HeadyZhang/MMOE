import tensorflow as tf
import numpy as np

class mmoe_v2:
    def __init__(self,
                 x_dim,
                 sparse_x_len,
                 emb_size,
                 layer_units=None,
                 layer_activations=None,
                 n_towers = 1,
                 tower_units=None,
                 tower_activations=None,
                 tower_weights='1',
                 fl_gamma=0.0,
                 fl_alpha=1.0,
                 alpha=0.5,
                 beta=0.2,
                 drm_coef=10,
                 base_coef=0.1,
                 lr=0.001,
                 logger=None,
                 bn_decay=0.9,
                 enable_drm=False,
                 use_sample_weight=False,
                 batch_norm=False,
                 iter_step=100,
                 base_model='fm',
                 enable_values=False
                 ):

        self.n_towers = n_towers
        self.logger = logger
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.lr = lr
        self.enable_drm = enable_drm
        self.batch_norm_decay = bn_decay
        self.batch_norm = batch_norm
        self.use_sample_weight = use_sample_weight
        self.layer_units = [int(x) for x in layer_units.split(',')] if type(layer_units) is not list \
            else layer_units

        self.layer_activations = list(map(lambda x: None if (x == 'None') else x, layer_activations.split(',')))  if type(layer_activations) is not list \
            else layer_activations

        self.tower_units = tower_units
        self.tower_activations = tower_activations
        if self.tower_units is not None and self.tower_activations is not None:
            assert len(self.tower_units) == len(self.tower_activations), 'layer units and activations length not match: %s vs %s' % (len(self.tower_units), len(self.tower_activations))

        self.tower_weights = [float(x) for x in tower_weights.split(',')]

        self.fl_alpha = fl_alpha
        self.fl_gamma = fl_gamma

        self.alpha = alpha
        self.beta = beta

        self.base_coef = base_coef
        self.drm_coef = drm_coef

        self.base_model = base_model

        self.x_indices = None
        self.x_values = None
        self.y = None
        self.w = None

        self.loss = None
        self.tower_sample_losses = None
        self.pred = None
        self.sample_weight = None

        self.loss_drm = None
        self.loss_base = None

        self.global_step = None

        self.batch_norm_layer = None
        self.train_op = None
        self.ordr1_weights = None
        self.ordr2_emb_mat = None
        self.every_n_iter = iter_step
        self.enable_values = enable_values


    def base_network(self, is_training):
        if self.base_model == 'fm':
            with tf.compat.v1.variable_scope('var/fm_first_order', reuse=tf.compat.v1.AUTO_REUSE):
                self.ordr1_weights = tf.compat.v1.get_variable('weight_matrix',
                                                               shape=[self.x_dim, 1],
                                                               dtype=tf.float32,
                                                               initializer=tf.initializers.random_normal(stddev=0.01))
                first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, self.x_indices)  # [None, sparse_x_len, 1]
                first_ordr = tf.reduce_sum(first_ordr, 2)  # [None, sparse_x_len], deepFM中用于拼接的一阶向量

                if self.enable_values:
                    first_ordr = first_ordr * self.x_values

                intersect = tf.compat.v1.get_variable('intersect', shape=[1], dtype=tf.float32,
                                                      initializer=tf.zeros_initializer())
                # intersect重复batch_size次后进行拼接
                intersect = tf.compat.v1.tile(intersect[tf.newaxis, :], [tf.compat.v1.shape(first_ordr)[0], 1])  # [None, 1]
                first_ordr = tf.concat([first_ordr, intersect], axis=1)  # [None, sparse_x_len + 1] 截距项也加入一阶向量

            with tf.compat.v1.variable_scope('var/fm_second_order', reuse=tf.compat.v1.AUTO_REUSE):
                self.ordr2_emb_mat = tf.compat.v1.get_variable('embedding_matrix',
                                                               shape=[self.x_dim, self.emb_size],
                                                               dtype=tf.float32,
                                                               initializer=tf.initializers.random_normal(stddev=0.01))
                # 使用最大的index来作为空值
                # ordr2_emb_mat = tf.concat([self.ordr2_emb_mat, tf.zeros(shape=[1, self.emb_size])],
                #                           axis=0)  # [x_dim + 1, emb_size]
                dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, self.x_indices)  # [None, sparse_x_len, emb_size]
                # sum -> sqr part
                if self.enable_values:
                    sum_dis_embs = tf.reduce_sum(dis_embs * tf.expand_dims(self.x_values, axis=2), 1)
                else:
                    sum_dis_embs = tf.reduce_sum(dis_embs, 1)  # [None, emb_size]
                sum_sqr_dis_embs = tf.square(sum_dis_embs)  # [None, emb_size]
                # sqr -> sum part
                sqr_dis_embs = tf.square(dis_embs)  # [None, sparse_x_len, emb_size]
                sqr_sum_dis_embs = tf.reduce_sum(sqr_dis_embs, 1)  # [None, emb_size]
                # second order
                second_ordr = 0.5 * (sum_sqr_dis_embs - sqr_sum_dis_embs)  # [None, emb_size]

            out_tensors = [first_ordr, second_ordr]  # shape = [None, sparse_x_len + 1 + emb_size]
            outputs = tf.concat(out_tensors, 1)
            return self.build_mlp_network(outputs, self.layer_units, self.layer_activations, is_training)
        else:
            raise TypeError('不支持的base_model类型: %s' % self.base_model)

    def build_mlp_network(self, input_tensor, layer_units, layer_activations, is_train, disable_last_bn=False):
        input_x = input_tensor
        for i, (units, activation) in enumerate(zip(layer_units, layer_activations)):
            input_x = tf.keras.layers.Dense(units, activation=activation)(input_x)
            if self.batch_norm:
                if disable_last_bn and i == len(layer_units) - 1:
                    continue
                input_x = tf.keras.layers.BatchNormalization(trainable=is_train)(input_x)

        return input_x

    def tower_network(self, base_output, is_train):
        with tf.compat.v1.variable_scope('var/tower_networks', reuse=tf.compat.v1.AUTO_REUSE):
            towers = []
            for i in range(self.n_towers):
                if self.tower_units is None or self.tower_activations is None:
                    # 当不设定tower网络参数时, 直接将输入进来的tensor进行reduce_sum后输出(兼容fm)
                    print('no tower network')
                    print(base_output.shape)
                    print(tf.reduce_sum(base_output, axis=1, keepdims=True).shape)
                    towers.append(tf.reduce_sum(base_output, axis=1, keepdims=True))
                else:
                    # 若base_output为单一tensor, 则每个tower都直接使用这一tensor; 若为list, 则每个tower选取使用对应的tensor
                    tower_input = base_output if type(base_output) is not list else base_output[i]
                    tower_out = self.build_mlp_network(tower_input, self.tower_units, self.tower_activations, is_train, True)
                    towers.append(tower_out)
            return tf.concat(towers, 1, name='tower_outputs')


    def forward(self, is_train=True):
        base_output = self.base_network(is_train)
        return  self.tower_network(base_output, is_train)

    def drm(self, y_true, is_treat, y_pred, with_th=True):
        msk_1 = tf.cast(is_treat, tf.bool)
        msk_0 = tf.cast(1 - is_treat, tf.bool)
        y_true_1 = tf.boolean_mask(y_true, msk_1, axis=0)
        y_true_0 = tf.boolean_mask(y_true, msk_0, axis=0)
        y_pred_1 = tf.boolean_mask(y_pred, msk_1, axis=0)
        y_pred_0 = tf.boolean_mask(y_pred, msk_0, axis=0)

        if with_th:
            y_1_score = tf.nn.softmax(tf.nn.tanh(y_pred_1), axis=0)
            y_0_score = tf.nn.softmax(tf.nn.tanh(y_pred_0), axis=0)
        else:
            y_1_score = tf.nn.softmax(y_pred_1, axis=0)
            y_0_score = tf.nn.softmax(y_pred_0, axis=0)

        return -tf.reduce_sum(y_1_score * y_true_1) + tf.reduce_sum(y_0_score * y_true_0)

    def focal_loss(self, y, p, gamma=0., alpha=1.):
        """
        基本思想: 在CE的基础上乘上一个调制因子factor, 对于CE LOSS更高的样本, 其factor更大. (即增加偏离label较大的样本的权重)
        :param y: label tensor, 只能取0或1, shape=(None,)
        :param p: predition tensor, shape=(None,)
        :param gamma: focusing parameter, 大于等于0, 取0时相当于不进行focus
        :param alpha: 用于正负样本均衡, 暂不实现
        :return:
        """
        _epsilon = 1e-7
        p = tf.clip_by_value(p, _epsilon, 1 - _epsilon)  # 限制概率区间, 防止后续出现log(0)的情况
        pt = y * p + (1 - y) * (1 - p)  # pt = p if y == 1 else 1 - p
        ce = -tf.compat.v1.log(pt)  # cross entropy
        factor = tf.pow(1 - pt, gamma)  # 调制因子, 由于0 < 1-pt < 1, 故当pt为常量时(1-pt)^gamma为递减函数, 即gamma越靠近0, 不确定样本的权重越高(越接近1)
        return ce * factor

    def cal_loss(self):

        tower_sample_losses = self.focal_loss(
            self.y,
            self.pred
        ) * tf.constant(self.tower_weights)  # shape=[None, n_towers]
        if self.sample_weight is not None:
            tower_sample_losses *= self.sample_weight  # shape=[None, n_towers]

        sample_loss = tf.reduce_sum(tower_sample_losses, axis=1)  # shape=[None,]
        sample_loss = tf.reduce_mean(sample_loss)

        return sample_loss, tower_sample_losses

    def print(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    ## 在干预下 曝光和带图率增量高
    def build_model(self, x_indices=None, x_values=None, y=None, w=None, mode=None):
        self.mode = mode

        with tf.compat.v1.variable_scope('input', reuse=tf.compat.v1.AUTO_REUSE):
            self.x_indices = tf.reshape(x_indices, shape=(-1, self.sparse_x_len))

            if self.enable_values:
                self.x_values = tf.reshape(x_values, shape=(-1, self.sparse_x_len))
                self.x_values = tf.concat([self.x_values[:, 0: 1] / 100, self.x_values[:, 1:]], axis=1)  # 分转元

            if y is not None:
                self.y = tf.cast(tf.reshape(y, (-1, self.n_towers)), dtype=tf.float32)

        with tf.compat.v1.variable_scope('train', reuse=tf.compat.v1.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.pred = self.forward(is_train=True)
            else:
                self.pred = self.forward(is_train=False)
            self.pred = tf.reshape(self.pred, (-1, self.n_towers))

            if mode == tf.estimator.ModeKeys.PREDICT:
                out_dict = {'prediction': self.pred}

                export_outputs = {
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(out_dict)
                }

                return tf.estimator.EstimatorSpec(mode=mode,
                                                  predictions=out_dict,
                                                  export_outputs=export_outputs)

            self.sample_weight = w if self.use_sample_weight else None
            self.loss, self.tower_sample_losses = self.cal_loss()
            ## 俩个label y0, y1

            tf.compat.v1.summary.scalar('loss', self.loss)

            eval_metric_ops = {

            }

            tensors_to_log = {
                "loss": self.loss
            }

            for i in range(self.n_towers):
                eval_metric_ops['tower_%s_loss' % i] =  tf.compat.v1.metrics.mean(self.tower_sample_losses[:, i])
                eval_metric_ops['tower_%s_auc' % i] = tf.compat.v1.metrics.auc(self.y[:, i], self.pred[:, i])
                tensors_to_log['tower_%s_auc' % i] =  eval_metric_ops['tower_%s_auc' % i][1]

            logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=self.every_n_iter)
            hooks = [logging_hook, tf.estimator.StepCounterHook()]
                # 训练过程
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.lr)
                grads, variables = zip(*optimizer.compute_gradients(self.loss))
                grads, global_norm = tf.clip_by_global_norm(grads, 5)
                self.train_op = optimizer.apply_gradients(zip(grads, variables),
                                                     global_step=tf.compat.v1.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op, training_hooks=hooks)


            elif mode == tf.estimator.ModeKeys.EVAL:

                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=hooks)
            else:
                raise ValueError('invalid mode: %s' % mode)


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    n_towers = 2
    target_weight = tf.constant([0.5, 0.5])  # shape=(3,)

    tower_units = [32, 32, 1]
    tower_activations = ['relu', 'relu', 'sigmoid']

    layer_units = [32, 32]
    layer_activations = ['relu', 'relu']
    batch_size, x_dim, sparse_x_len, emb_size = 3, 100, 20, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    data_y = np.random.random_integers(0, 1, size=(batch_size, n_towers))

    data_w = np.random.standard_normal(size=(batch_size, n_towers))

    placeholder_x = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, n_towers))
    placeholder_w = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, n_towers))

    model = mmoe_v2(x_dim,
                      sparse_x_len,
                      emb_size,
                      layer_units=layer_units,
                      layer_activations=layer_activations,
                      n_towers=n_towers,
                      tower_units=tower_units,
                      tower_activations=tower_activations,
                      tower_weights='0.5,0.5',
                      fl_gamma=0.0,
                      fl_alpha=1.0,
                      alpha=0.5,
                      beta=0.2,
                      drm_coef=10,
                      base_coef=0.1,
                      lr=0.001,
                      logger=None,
                      bn_decay=0.9,
                      enable_drm=False,
                      use_sample_weight=True,
                      batch_norm=True)

    model.build_model(x=placeholder_x, y=placeholder_y, w=placeholder_w, mode=tf.estimator.ModeKeys.TRAIN)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        for _ in range(100):
            _, loss, pred = sess.run([model.train_op, model.loss, model.pred],
                                       feed_dict={model.x: data_x,
                                                  model.y: data_y,
                                                  model.sample_weight: data_w
                                                  })
            print(loss)
        print(pred)







