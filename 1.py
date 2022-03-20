import numpy as np
import jieba
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, LSTM, Embedding, Dense


def decode_sequence(input_seq):
    # encoder_states = [state_h, state_c]
    states_value = encoder_model.predict(input_seq)  # list 2个 array 1*rnn_size
    target_seq = np.zeros((1, 1))
    # 目标输入序列 初始为 'BEGIN_' 的 idx
    target_seq[0, 0] = outputToken_idx['BEGIN_']
    stop = False
    decoded_sentence = ''
    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # output_tokens [1*1*9126]   h,c [1*rnn_size]
        sampled_token_idx = np.argmax(output_tokens)
        sampled_word = idx_outputToken[sampled_token_idx]
        decoded_sentence += ' ' + sampled_word

        if sampled_word == '_END' or len(decoded_sentence) > 60:
            stop = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_idx  # 作为下一次预测，输入
        # Update states
        states_value = [h, c]  # 作为下一次的状态输入

    return decoded_sentence


if __name__ == '__main__':
    input_texts = []  # 输入句子集
    target_texts = []  # 输出句子集
    input_words = set()  # 输入词集合
    target_words = set()  # 输出词集合

    with open('cmn.txt', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for i in range(len(lines) - 22500):
        x, y, z = lines[i].split('\t')
        x = x[:-1]
        input_texts.append(x)
        y = y[:-1]
        y = jieba.lcut(y)
        y = ' '.join(y)
        y = 'BEGIN_ ' + y + ' _END'  # 添加起始符和结束符
        target_texts.append(y)
        for word in x.split():
            if word not in input_words:
                input_words.add(word)
        for word in y.split():
            if word not in target_words:
                target_words.add(word)

    max_input_seq_len = max([len(seq.split())
                             for seq in input_texts])  # 输入句子最大长度
    max_target_seq_len = max([len(seq.split())
                              for seq in target_texts])  # 输出句子最大长度
    input_words = sorted(list(input_words))  # 排序输入词语(对建立映射有用)
    target_words = sorted(list(target_words))
    num_encoder_tokens = len(input_words)
    num_decoder_tokens = len(target_words)

    # 建立映射关系
    inputToken_idx = {token: i for (i, token) in enumerate(input_words)}
    outputToken_idx = {token: i for (i, token) in enumerate(target_words)}
    idx_inputToken = {i: token for (i, token) in enumerate(input_words)}
    idx_outputToken = {i: token for (i, token) in enumerate(target_words)}
    # 训练输入维度
    encoder_input_data = np.zeros(
        (len(input_texts), max_input_seq_len),
        # 句子数量，         最大输入句子长度
        dtype=np.float32)
    decoder_input_data = np.zeros(
        (len(target_texts), max_target_seq_len),
        # 句子数量，          最大输出句子长度
        dtype=np.float32)
    decoder_output_data = np.zeros(
        (len(target_texts), max_target_seq_len, num_decoder_tokens),
        # 句子数量，          最大输出句子长度,      输出 tokens ids 个数
        dtype=np.float32)
    for i, (input_text,
            target_text) in enumerate(zip(input_texts, target_texts)):
        for t, word in enumerate(input_text.split()):
            encoder_input_data[i, t] = inputToken_idx[word]
        for t, word in enumerate(target_text.split()):
            decoder_input_data[i, t] = outputToken_idx[word]
            if t > 0:
                # 解码器的输出比输入提前一个时间步
                decoder_output_data[i, t - 1, outputToken_idx[word]] = 1.

    embedding_size = 256  # 嵌入维度
    rnn_size = 64
    # 编码器
    encoder_inputs = Input(shape=(None, ))
    encoder_after_embedding = Embedding(
        input_dim=num_encoder_tokens,  # 输入单词个数
        output_dim=embedding_size)(encoder_inputs)
    encoder_lstm = LSTM(units=rnn_size, return_state=True)
    # return_state: Boolean. Whether to return
    #   the last state in addition to the output.
    _, state_h, state_c = encoder_lstm(encoder_after_embedding)
    encoder_states = [state_h, state_c]  # 思想向量

    # 解码器
    decoder_inputs = Input(shape=(None, ))
    decoder_after_embedding = Embedding(
        input_dim=num_decoder_tokens,  # 输出单词个数
        output_dim=embedding_size)(decoder_inputs)
    decoder_lstm = LSTM(units=rnn_size,
                        return_sequences=True,
                        return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_after_embedding,
                                         initial_state=encoder_states)
    # 使用 encoder 输出的思想向量初始化 decoder 的 LSTM 的初始状态
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    # 输出词个数,多分类
    decoder_outputs = decoder_dense(decoder_outputs)
    # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(x=[encoder_input_data, decoder_input_data],
    #           y=decoder_output_data,
    #           batch_size=128,
    #           epochs=300,
    #           validation_split=0.1)
    # model.save('model.h5')
    # model = load_model('model.h5')
    encoder_model = Model(encoder_inputs,
                          encoder_states)  # 输入（带embedding），输出思想向量
    # 编码器的输出，作为解码器的初始状态
    decoder_state_input_h = Input(shape=(rnn_size, ))
    decoder_state_input_c = Input(shape=(rnn_size, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
        decoder_after_embedding, initial_state=decoder_states_inputs)
    # 作为下一次推理的状态输入 h, c
    decoder_states_inf = [state_h_inf, state_c_inf]
    # LSTM的输出，接 FC，预测下一个词是什么
    decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs_inf] + decoder_states_inf)

    # 简单测试 采样

    # text_to_translate = 'I hate you'
    text_to_translate = input_texts[0:20]
    for i in range(len(text_to_translate)):
        encoder_input_to_translate = np.zeros((1, max_input_seq_len),
                                              dtype=np.float32)
        for t, word in enumerate(text_to_translate[i].split()):
            encoder_input_to_translate[0, t] = inputToken_idx[word]
        print(text_to_translate[i])
        # encoder_input_to_translate [[ids,...,0,0,0,0]]
        print(decode_sequence(encoder_input_to_translate))
