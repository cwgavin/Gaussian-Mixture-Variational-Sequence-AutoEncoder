from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def auc_score(y_true, y_score):
    precision, recall, _ = precision_recall_curve(1-y_true, 1-y_score)
    return auc(recall, precision)


def filling_batch(batch_data, args):  # zero-pad batch to meet batch size
    new_batch_data = []
    last_batch_size = len(batch_data[0])
    for b in batch_data:
        new_batch_data.append(
            np.concatenate([b, [np.zeros_like(b[0]).tolist()
                                for _ in range(args.batch_size - last_batch_size)]], axis=0))
    return new_batch_data


def compute_likelihood(sess, model, sampler, purpose, args):
    all_likelihood = []
    for batch_data in sampler.iterate_all_data(args.batch_size,
                                               partial_ratio=args.partial_ratio,
                                               purpose=purpose):
        if len(batch_data[0]) < args.batch_size:
            last_batch_size = len(batch_data[0])
            batch_data = filling_batch(batch_data, args)
            feed = dict(zip(model.input_form, batch_data))
            batch_likelihood = sess.run(model.batch_likelihood, feed)[:last_batch_size]
        else:
            feed = dict(zip(model.input_form, batch_data))
            batch_likelihood = sess.run(model.batch_likelihood, feed)
        all_likelihood.append(batch_likelihood)
    return np.concatenate(all_likelihood)


def compute_loss(sess, model, sampler, purpose, args):
    all_loss = []
    if hasattr(args, 'pt') and args.pt:
        loss_op = model.pretrain_loss
    else:
        loss_op = model.loss  # for SAE and VSAE (no pt option)
    for batch_data in sampler.iterate_all_data(args.batch_size,
                                               partial_ratio=args.partial_ratio,
                                               purpose=purpose):
        if len(batch_data[0]) < args.batch_size:
            batch_data = filling_batch(batch_data, args)
            feed = dict(zip(model.input_form, batch_data))
            loss = sess.run(loss_op, feed)
        else:
            feed = dict(zip(model.input_form, batch_data))
            loss = sess.run(loss_op, feed)
        all_loss.append(loss)
    return np.mean(all_loss)
