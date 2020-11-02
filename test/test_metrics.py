from libs.targets_transformers import *
from libs.metrics import *

def test_accuracy_eq():
    for model_type in ['PC', 'OR', 'ORbin']:
        bs_values = [s for s in np.random.randint(1, 128, 32)]
        bs_values = [1] + bs_values
        for bs in bs_values:
            metric = accuracy(model_type=model_type, batch_size=bs)
            transformer = targets_transformer(model_type=model_type, batch_size=bs)
            y_true = np.random.choice(np.arange(0, 9, 1).astype(np.int32), bs)
            # y_true_t = t.from_numpy(y_true).cuda()
            y_true_transformed = transformer(y_true)
            y_true_transformed_noised = y_true_transformed + t.randn_like(y_true_transformed.float())/100.0
            y_true_transformed_noised = t.clamp(y_true_transformed_noised, 0.0, 1.0)

            assert np.isclose(metric(y_true_transformed_noised, y_true_transformed).cpu().numpy().mean(), 1.0)



def test_accuracy_leq():
    for model_type in ['PC', 'OR', 'ORbin']:
        bs_values = [s for s in np.random.randint(1, 128, 32)]
        bs_values = [1] + bs_values
        for bs in bs_values:
            metric = accuracy(model_type=model_type, batch_size=bs)
            transformer = targets_transformer(model_type=model_type, batch_size=bs)
            y_true = np.random.choice(np.arange(0, 9, 1).astype(np.int32), bs)
            # y_true_t = t.from_numpy(y_true).cuda()
            y_true_transformed = transformer(y_true)

            y_pred = y_true = np.random.choice(np.arange(0, 9, 1).astype(np.int32), bs)
            # y_pred_t = t.from_numpy(y_pred).cuda()
            y_pred_transformed = transformer(y_pred)

            assert metric(y_true_transformed, y_pred_transformed).cpu().numpy() <= 1.0
            assert metric(y_true_transformed, y_pred_transformed).cpu().numpy() >= 0.0