# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.tf_utils import *
from functools import reduce



class MMOE:

    def __init__(self, x_dim, sparse_x_len, emb_size, tower_units=None, tower_activations=None, lr=0.001,
                 batch_norm=False, bn_decay=0.9, alpha=1.8, beta=0.5,
                 nn_init='glorot_uniform', logger=None, use_sample_weight=False,
                 fl_gamma=0., fl_alpha=1., base_model='fm', n_towers=1, tower_weights='1',
                 expert_units=None, expert_activations=None, n_experts=None, infer_weights=None,
                 export_original_outputs=False, label_names=None, num_levels=None, keep_pro=None,
                 iter_step=100):
        """

        :param x_dim:
        :param sparse_x_len:
        :param emb_size: fm embedding size
        :param tower_units: tower nn units
        :param tower_activations: deepfm nn activations
        :param lr: learning rate
        :param batch_norm: 是否进行batch normalization
        :param bn_decay: batch normalization decay参数
        :param alpha: 正则化参数, 总权重
        :param beta: 正则化参数, L1/L2占比
        :param nn_init: nn参数初始化方法
        :param logger:
        :param use_sample_weight: 是否使用sample weight
        :param fl_gamma: focal loss的gamma值
        :param fl_alpha: focal loss的alpha值
        :param base_model: 底层模型种类, 如fm, moe, mmoe (moe, mmoe时专家相关参数才生效)
        :param n_towers
        :param tower_weights
        :param expert_units: 专家网络神经元个数
        :param expert_activations: 专家网络激活函数
        :param n_experts: 专家网络个数
        :param infer_weights: 推断时的tower weight
        :param export_original_outputs: 导出模型时是否以所有目标原始值的形式导出
        :param id_col_configs:
        :param vector_col_dim:
        """
        # logger
        self.logger = logger

        # 模型参数设置
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.tower_units = tower_units
        self.tower_activations = tower_activations
        if self.tower_units is not None and self.tower_activations is not None:
            assert len(self.tower_units) == len(self.tower_activations), 'layer units and activations length not match: %s vs %s' % (len(self.tower_units), len(self.tower_activations))
        self.lr = lr
        self.batch_norm = batch_norm
        self.batch_norm_decay = bn_decay
        self.alpha = alpha
        self.beta = beta
        self.nn_init = nn_init
        self.use_sample_weight = use_sample_weight
        self.mode = None
        self.expert_units = expert_units
        self.expert_activations = expert_activations
        self.n_experts = n_experts

        self.indices, self.values = [], []
        self.fl_gamma = fl_gamma
        self.fl_alpha = fl_alpha
        self.base_model = base_model
        self.n_towers = n_towers
        self.tower_weights = [float(x) for x in tower_weights.split(',')]
        self.infer_weights = self.tower_weights if infer_weights is None else [float(x) for x in infer_weights.split(',')]
        self.export_original_outputs = export_original_outputs
        self.keep_pro = keep_pro ## drop out

        # PLE
        self.label_names = label_names
        self.num_levels = num_levels

        # 模型重要tensor声明
        self.x = None
        self.y = None
        self.pred = None
        self.weighted_pred = None
        self.loss = None
        self.global_step = None
        self.train_op = None
        self.ordr1_weights = None
        self.ordr2_emb_mat = None
        self.sample_weight = None
        self.tower_sample_losses = None
        self.reg_loss = None
        self.param_num = None
        self.summary = None
        self.every_n_iter = iter_step

    def build_mlp_network(self, input_tensor, layer_units, layer_activations, is_train, disable_last_bn=False):
        input_x = input_tensor
        for i, (units, activation) in enumerate(zip(layer_units, layer_activations)):
            input_x = tf.keras.layers.Dense(units, activation=activation)(input_x)
            if self.batch_norm:
                if disable_last_bn and i == len(layer_units) - 1:
                    continue
                input_x = tf.keras.layers.BatchNormalization(trainable=is_train)(input_x)

        return input_x

    def base_network(self, x, is_train):
        """
        构建多目标模型的底层共享网络
        :param x:
        :param is_train:
        :return: base output tensor
        """
        # 定义batch_norm layer

        if self.base_model == 'fm':
            # fm前向过程
            # factorization machine只支持二分类
            with tf.compat.v1.variable_scope('var/fm_first_order', reuse=tf.compat.v1.AUTO_REUSE):
                self.ordr1_weights = tf.compat.v1.get_variable('weight_matrix',
                                                     shape=[self.x_dim, 1],
                                                     dtype=tf.float32,
                                                     initializer=tf.initializers.random_normal(stddev=0.01))
                # # 使用最大的index来作为空值
                # ordr1_weights = tf.concat([self.ordr1_weights, tf.zeros(shape=[1, 1])], axis=0)  # [x_dim + 1, 1]
                # ^^其实没必要, 当lookup的id大于lookup param维度时, tf会直接返回0

                # 查找每个离散值index对应维度的order1权重
                first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, x)  # [None, sparse_x_len, 1]
                first_ordr = tf.reduce_sum(first_ordr, 2)   # [None, sparse_x_len], deepFM中用于拼接的一阶向量

                intersect = tf.compat.v1.get_variable('intersect', shape=[1], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
                # intersect重复batch_size次后进行拼接
                intersect = tf.tile(intersect[tf.newaxis, :], [tf.shape(first_ordr)[0], 1])     # [None, 1]
                first_ordr = tf.concat([first_ordr, intersect], axis=1)     # [None, sparse_x_len + 1] 截距项也加入一阶向量

            with tf.compat.v1.variable_scope('var/fm_second_order', reuse=tf.compat.v1.AUTO_REUSE):
                self.ordr2_emb_mat = tf.compat.v1.get_variable('embedding_matrix',
                                                     shape=[self.x_dim, self.emb_size],
                                                     dtype=tf.float32,
                                                     initializer=tf.initializers.random_normal(stddev=0.01))
                # 使用最大的index来作为空值
                # ordr2_emb_mat = tf.concat([self.ordr2_emb_mat, tf.zeros(shape=[1, self.emb_size])],
                #                           axis=0)  # [x_dim + 1, emb_size]
                dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, x)  # [None, sparse_x_len, emb_size]
                # sum -> sqr part
                sum_dis_embs = tf.reduce_sum(dis_embs, 1)  # [None, emb_size]
                sum_sqr_dis_embs = tf.square(sum_dis_embs)  # [None, emb_size]
                # sqr -> sum part
                sqr_dis_embs = tf.square(dis_embs)  # [None, sparse_x_len, emb_size]
                sqr_sum_dis_embs = tf.reduce_sum(sqr_dis_embs, 1)  # [None, emb_size]
                # second order
                second_ordr = 0.5 * (sum_sqr_dis_embs - sqr_sum_dis_embs)  # [None, emb_size]

            out_tensors = [first_ordr, second_ordr]      # shape = [None, sparse_x_len + 1 + emb_size]
            return tf.concat(out_tensors, 1)
        elif self.base_model in ('moe', 'mmoe'):
            # 对于moe, 返回tower间共用的一个专家输出tensor
            # 对于mmoe, 返回每个tower自定义的专家输出tensor list
            with tf.compat.v1.variable_scope('var/base_embedding', reuse=tf.compat.v1.AUTO_REUSE):
                self.ordr2_emb_mat = tf.compat.v1.get_variable('embedding_matrix',
                                                     shape=[self.x_dim, self.emb_size],
                                                     dtype=tf.float32,
                                                     initializer=tf.initializers.random_normal(stddev=0.01))
                dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, x)  # [None, sparse_x_len, emb_size]
                # 通过embedding将输入维度从x_dim降至sparse_x_len * emb_size
                dense_x = [tf.reshape(dis_embs, shape=(-1, self.sparse_x_len * self.emb_size))]

                dense_x = tf.concat(dense_x, 1)

            with tf.compat.v1.variable_scope('var/expert_networks', reuse=tf.compat.v1.AUTO_REUSE):
                # 构建专家网络
                experts = []
                for i in range(self.n_experts):
                    # 专家网络输出向量shape=(None, expert_units[-1])
                    out = self.build_mlp_network(
                        dense_x,
                        self.expert_units,
                        self.expert_activations,
                        is_train,
                        disable_last_bn=False
                    )
                    out = out[:, tf.newaxis, :]                             # shape=(None, 1, expert_units[-1])
                    experts.append(out)
                experts = tf.concat(experts, 1, name='expert_origin_outputs')      # shape=(None, n_experts, expert_units[-1])

                # 专家控制门
                if self.base_model == 'moe':
                    # 对于moe, 所有tower共享一个gate
                    gate_kernel = tf.compat.v1.get_variable('expert_gate',
                                                  shape=[self.sparse_x_len * self.emb_size, self.n_experts],
                                                  dtype=tf.float32,
                                                  initializer=tf.initializers.random_normal(stddev=0.01))
                    gate = tf.matmul(dense_x, gate_kernel)  # shape=(None, n_experts)
                    gate = tf.nn.softmax(gate)
                    gate = tf.tile(gate[:, :, tf.newaxis], [1, 1, self.expert_units[-1]])  # shape=(None, n_experts, expert_units[-1])

                    # 专家网络输出根据控制门进行加权得到最终输出
                    out = experts * gate                            # shape=(None, n_experts, expert_units[-1])
                    experts_final = tf.reduce_sum(out, axis=1)      # shape=(None, experts_units[-1])
                else:
                    # 对于mmoe, 每个tower都有其单独的gate和公共的专家网络进行加权, 循环多次为每个tower计算控制门
                    experts_final = []
                    for i in range(self.n_towers):
                        gate_kernel = tf.compat.v1.get_variable('expert_gate_%s' % i,
                                                      shape=[self.sparse_x_len * self.emb_size, self.n_experts],
                                                      dtype=tf.float32,
                                                      initializer=tf.initializers.random_normal(stddev=0.01))
                        gate = tf.matmul(dense_x, gate_kernel)                      # shape=(None, n_experts)
                        gate = tf.nn.softmax(gate)
                        gate = tf.tile(gate[:, :, tf.newaxis], [1, 1, self.expert_units[-1]])   # shape=(None, n_experts, expert_units[-1])

                        # 专家网络输出根据控制门进行加权得到最终输出
                        out = experts * gate                              # shape=(None, n_experts, expert_units[-1])
                        out = tf.reduce_sum(out, axis=1)                  # shape=(None, experts_units[-1])
                        experts_final.append(out)

                return experts_final
        elif self.base_model == 'ple':
            with tf.compat.v1.variable_scope('var/base_embedding', reuse=tf.compat.v1.AUTO_REUSE):
                self.ordr2_emb_mat = tf.compat.v1.get_variable('embedding_matrix',
                                                     shape=[self.x_dim, self.emb_size],
                                                     dtype=tf.float32,
                                                     initializer=tf.initializers.random_normal(stddev=0.01))
                dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, x)  # [None, sparse_x_len, emb_size]
                # 通过embedding将输入维度从x_dim降至sparse_x_len * emb_size
                dense_x = [tf.reshape(dis_embs, shape=(-1, self.sparse_x_len * self.emb_size))]
                print(tf.concat(dense_x, 1).shape)

                dense_x = tf.concat(dense_x, 1)

            with tf.compat.v1.variable_scope('var/extract_network', reuse=tf.compat.v1.AUTO_REUSE):
                outputs = dense_x
                for level in range(self.num_levels):
                    # 如果不是第一层，那么输入是多个上层的输出expert组成的列表
                    # 此时，需要进行fusion：一般是拼接、相乘、相加几种融合方式
                    # 这里使用相加拼接相乘
                    if isinstance(outputs, list):
                        outputs = tf.concat([reduce(lambda x, y: x + y, outputs),
                                             reduce(lambda x, y: x * y, outputs)],
                                            axis=-1)

                    with tf.compat.v1.variable_scope('Mixture-of-Experts', reuse=tf.compat.v1.AUTO_REUSE):
                        mixture_experts = []
                        for name in self.label_names + ["expert_shared"]:
                            for i in range(self.n_experts):
                                # expert一般是一层全连接层
                                out = self.build_mlp_network(
                                    input_tensor=outputs,
                                    layer_units=[self.expert_units[level]],
                                    layer_activations=[self.expert_activations[level]],
                                    is_train=is_train,
                                    disable_last_bn=False
                                )

                                # out = out[:, tf.newaxis, :]  # shape=(None, 1, expert_units[-1])
                                mixture_experts.append(out)
                    # 如果是最后一层，那么gate的数量应该是task的数量
                    # 其他层的话，gate的数量一般等于experts的数量
                    if level == self.num_levels - 1:
                        num_gates = len(self.label_names)
                    else:
                        num_gates = self.n_experts

                    # 生成不同专家任务的门
                    with tf.compat.v1.variable_scope("multi-gate", reuse=tf.compat.v1.AUTO_REUSE):
                        multi_gate = []
                        for i in range(num_gates):
                            gate = tf.keras.layers.Dense(dense_x, units=self.n_experts * (len(self.label_names) + 1),
                                                   kernel_initializer=tf.initializers.random_uniform(-0.01, 0.01),
                                                   bias_initializer=tf.initializers.zeros(),
                                                   name="gate_{}_level{}".format(i, level)
                                                   )
                            gate = tf.nn.softmax(gate)
                            multi_gate.append(gate)

                    with tf.compat.v1.variable_scope("combine_gate_expert", reuse=tf.compat.v1.AUTO_REUSE):
                        ple_layers = []
                        for i in range(num_gates):
                            ple_layers.append(self._combine_expert_gate(mixture_experts, multi_gate[i]))
                    outputs = ple_layers
                return outputs
        else:
            raise TypeError('不支持的base_model类型: %s' % self.base_model)

    def _combine_expert_gate(self, mixture_experts, gate):
        mixture_experts = tf.concat([tf.expand_dims(dnn, axis=1) for dnn in mixture_experts], axis=1)
        gate = tf.expand_dims(gate, axis=-1)
        return tf.reduce_sum(mixture_experts * gate, axis=-1)

    def tower_network(self, base_output, is_train):
        """
        构建各个目标的专用网络
        :param base_output: tensor or list<tensor>
        :param is_train:
        :return:
        """
        with tf.compat.v1.variable_scope('var/tower_networks', reuse=tf.compat.v1.AUTO_REUSE):
            # 定义batch_norm layer
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
                    tower_out = self.build_mlp_network(
                        tower_input,
                        self.tower_units,
                        self.tower_activations,
                        is_train,
                        disable_last_bn=True)
                    towers.append(tower_out)
            return tf.concat(towers, 1, name='tower_outputs')

    def forward(self, is_train):
        """

        :param is_train:
        :return:
        """
        # 对id类型的输入进行embedding

        base_output = self.base_network(self.x, is_train)
        return self.tower_network(base_output, is_train)

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
        # shape=[None, n_towers]
        # use focal loss
        tower_sample_losses = self.focal_loss(
            self.y,
            self.pred,
            gamma=self.fl_gamma,
            alpha=self.fl_alpha
        ) * tf.constant(self.tower_weights)                             # shape=[None, n_towers]
        if self.sample_weight is not None:
            tower_sample_losses *= self.sample_weight                   # shape=[None, n_towers]

        sample_loss = tf.reduce_sum(tower_sample_losses, axis=1)        # shape=[None,]
        sample_loss = tf.reduce_mean(sample_loss)                       # shape=[] (scalar)

        # 所有参数组成的向量，用以添加正则项

        return sample_loss, tower_sample_losses

    def build_model(self, x=None, y=None, w=None, mode=None, hooks=None):
        """

        :param x: tensor类型, 稀疏表达的multi-hot向量, 包含所有向量值为1的indices
        :param vector_x: tensor类型, embedding类型的特征
        :param id_map: dict类型, e.g. {'uid': tensor}, 需要进行embedding的id特征
        :param y: tensor类型, label
        :param w: tensor类型, sample weight
        :param mode: tf.estimator.ModeKeys的某个枚举值, 用于指定模型处于训练、评估或是推断模式
        :param hooks: TensorFlow hook
        :return:
        """
        self.mode = mode

        with tf.compat.v1.variable_scope('input', reuse=tf.compat.v1.AUTO_REUSE):
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))   # 任何情况下x都不能为空

            if y is not None:
                # 对于推理过程, y可以为空
                self.y = tf.cast(tf.reshape(y, (-1, self.n_towers)), dtype=tf.float32)

        with tf.compat.v1.variable_scope('train', reuse=tf.compat.v1.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.pred = self.forward(is_train=True)     # shape=[None, n_towers]
            else:
                self.pred = self.forward(is_train=False)    # shape=[None, n_towers]

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
                eval_metric_ops['tower_%s_loss' % i] = tf.compat.v1.metrics.mean(self.tower_sample_losses[:, i])
                eval_metric_ops['tower_%s_auc' % i] = tf.compat.v1.metrics.auc(self.y[:, i], self.pred[:, i])
                tensors_to_log['tower_%s_auc' % i] = eval_metric_ops['tower_%s_auc' % i][1]

            logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=self.every_n_iter)
            hooks = [logging_hook, tf.estimator.StepCounterHook()]
            # 训练过程
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.lr)
                grads, variables = zip(*optimizer.compute_gradients(self.loss))
                grads, global_norm = tf.clip_by_global_norm(grads, 5)
                self.train_op = optimizer.apply_gradients(zip(grads, variables),
                                                          global_step=tf.compat.v1.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op,
                                                  training_hooks=hooks)


            elif mode == tf.estimator.ModeKeys.EVAL:

                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops,
                                                  evaluation_hooks=hooks)
            else:
                raise ValueError('invalid mode: %s' % mode)

    def print(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)


if __name__ == '__main__':
    # 可直接运行, 测试类的逻辑正确性
    # mock data
    tf.compat.v1.disable_eager_execution()
    batch_size, x_dim, sparse_x_len, emb_size = 3, 100, 10, 5
    n_towers = 3
    target_weight = tf.constant([0.5, 0.2, 0.3])  # shape=(3,)

    tower_units = [32, 32, 1]
    tower_activations = ['relu', 'relu', 'sigmoid']
    # tower_units = None
    # tower_activations = None

    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))
    data_y = np.random.random_integers(0, 1, size=(batch_size, n_towers))
    data_w = np.random.standard_normal(size=(batch_size, n_towers))

    vector_col_dim = 5
    data_vector_x = np.random.standard_normal(size=(batch_size, vector_col_dim))

    # place holders
    placeholder_x = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, n_towers))
    placeholder_w = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, n_towers))


    # placeholder_id = None
    # id_col_configs = None

    # model init
    model = MMOE(x_dim, sparse_x_len, emb_size,
                                tower_units=tower_units, tower_activations=tower_activations, lr=0.01,
                                batch_norm=True, use_sample_weight=True, base_model='fm', n_towers=n_towers,
                                tower_weights='0.5,0.2,0.3',
                                expert_units=[16, 16], expert_activations=['relu', 'relu'], n_experts=4)
    model.build_model(x=placeholder_x, y=placeholder_y, w=placeholder_w, mode=tf.estimator.ModeKeys.TRAIN)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        for _ in range(100):
            _, loss = sess.run([model.train_op, model.loss],
                               feed_dict={model.x: data_x,
                                          model.y: data_y,
                                          model.sample_weight: data_w,
                                          })
            print(loss)
        print(sess.run(model.pred, feed_dict={model.x: data_x}))
