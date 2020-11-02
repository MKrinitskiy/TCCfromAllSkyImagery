from libs.targets_transformers import targets_transformer, targets_transformer_np
import numpy as np
import torch as t

def test_transform_and_back():
    for model_type in ['PC', 'OR', 'ORbin']:
        bs_values = [s for s in np.random.randint(2, 128, 8)]
        bs_values = [1] + bs_values
        for bs in bs_values:
            K_values = [s for s in np.random.randint(3, 32, 8)]
            for K in K_values:
                transformer = targets_transformer(model_type=model_type, batch_size=bs, classes_number=K)
                y_true = np.random.choice(np.arange(0, K, 1).astype(np.int32), bs)
                y_true_t = t.from_numpy(y_true)
                y_true_transformed = transformer(y_true)
                y_true_transformed_back = transformer.transform_back(y_true_transformed).cpu()
                assert t.all(t.eq(y_true_transformed_back, y_true_t)).numpy()


def test_transform_PC():
    K_values = [s for s in np.random.randint(4, 32, 8)]
    bs = 4
    for K in K_values:
        transformer = targets_transformer(model_type='PC', batch_size=bs, classes_number=K)
        y_true = np.array([0, 1, K-2, K-1]).astype(np.int32)
        y_true_transformed = transformer(y_true).cpu().numpy()
        assert ((y_true_transformed[0,0] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[1, 1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[2, -2] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[3, -1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))


def test_transform_ORbin():
    K_values = [s for s in np.random.randint(4, 32, 8)]
    bs = 4
    for K in K_values:
        transformer = targets_transformer(model_type='ORbin', batch_size=bs, classes_number=K)
        y_true = np.array([0, 1, K-2, K-1]).astype(np.int32)
        y_true_transformed = transformer(y_true).cpu().numpy()
        assert ((y_true_transformed[0,0] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[1, 1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[2, -2] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[3, -1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))


def test_transform_np_OR():
    K_values = [s for s in np.random.randint(4, 32, 8)]
    bs_values = [s for s in np.random.randint(2, 128, 32)]
    bs_values = [1] + bs_values
    for bs in bs_values:
        for K in K_values:
            transformer = targets_transformer(model_type='OR', batch_size=bs, classes_number=K)
            y_true = np.random.choice(np.arange(0, K, 1).astype(np.int32), bs)
            y_true_transformed = transformer(y_true).cpu().numpy()
            for row,yt in zip(range(bs), y_true):
                yt_row = y_true_transformed[row, :]
                assert np.allclose(yt_row[:yt + 1], np.ones_like(yt_row[:yt + 1]))
                assert np.allclose(yt_row[yt + 1:], np.zeros_like(yt_row[yt + 1:]))


def test_transform_and_back_float():
    for model_type in ['PC', 'OR', 'ORbin']:
        bs_values = [s for s in np.random.randint(2, 128, 32)]
        bs_values = [1] + bs_values
        for bs in bs_values:
            K_values = [s for s in np.random.randint(3, 32, 8)]
            for K in K_values:
                transformer = targets_transformer(model_type=model_type, batch_size=bs, classes_number=K)
                y_true = np.random.choice(np.arange(0, K, 1).astype(np.int32), bs)
                y_true_t = t.from_numpy(y_true)
                y_true_transformed = transformer(y_true)
                y_true_transformed_noised = y_true_transformed.float() + t.randn_like(y_true_transformed.float()) / 20.0
                y_true_transformed_noised = t.clamp(y_true_transformed_noised, 0.0, 1.0)
                y_true_transformed_back = transformer.transform_back(y_true_transformed_noised).cpu()
                assert t.all(t.eq(y_true_transformed_back, y_true_t)).numpy()


def test_transform_np_and_back():
    for model_type in ['PC', 'OR', 'ORbin']:
        bs_values = [s for s in np.random.randint(2, 128, 32)]
        bs_values = [1] + bs_values
        for bs in bs_values:
            K_values = [s for s in np.random.randint(3, 32, 8)]
            for K in K_values:
                transformer = targets_transformer_np(model_type=model_type, batch_size=bs, classes_number=K)
                y_true = np.random.choice(np.arange(0, K, 1).astype(np.int32), bs)
                y_true_transformed = transformer(y_true)
                y_true_transformed_back = transformer.transform_back(y_true_transformed)
                assert np.alltrue(y_true_transformed_back == y_true)


def test_transform_np_PC():
    K_values = [s for s in np.random.randint(4, 32, 8)]
    bs = 4
    for K in K_values:
        transformer = targets_transformer_np(model_type='PC', batch_size=bs, classes_number=K)
        y_true = np.array([0, 1, K-2, K-1])
        y_true_transformed = transformer(y_true)
        assert ((y_true_transformed[0,0] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[1, 1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[2, -2] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[3, -1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))



def test_transform_np_ORbin():
    K_values = [s for s in np.random.randint(4, 32, 8)]
    bs = 4
    for K in K_values:
        transformer = targets_transformer_np(model_type='ORbin', batch_size=bs, classes_number=K)
        y_true = np.array([0, 1, K-2, K-1])
        y_true_transformed = transformer(y_true)
        assert ((y_true_transformed[0,0] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[1, 1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[2, -2] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))
        assert ((y_true_transformed[3, -1] == 1) & (np.square(y_true_transformed[0, :]).sum() == 1.0))


def test_transform_np_OR():
    K_values = [s for s in np.random.randint(4, 32, 8)]
    bs_values = [s for s in np.random.randint(2, 128, 32)]
    bs_values = [1] + bs_values
    for bs in bs_values:
        for K in K_values:
            transformer = targets_transformer_np(model_type='OR', batch_size=bs, classes_number=K)
            y_true = np.random.choice(np.arange(0, K, 1).astype(np.int32), bs)
            y_true_transformed = transformer(y_true)
            for row,yt in zip(range(bs), y_true):
                yt_row = y_true_transformed[row, :]
                assert np.allclose(yt_row[:yt + 1], np.ones_like(yt_row[:yt + 1]))
                assert np.allclose(yt_row[yt + 1:], np.zeros_like(yt_row[yt + 1:]))


def test_transform_np_and_back_float():
    for model_type in ['PC', 'OR', 'ORbin']:
        bs_values = [s for s in np.random.randint(1, 128, 32)]
        bs_values = [1] + bs_values
        for bs in bs_values:
            transformer = targets_transformer_np(model_type=model_type, batch_size=bs)
            y_true = np.random.choice(np.arange(0, 9, 1).astype(np.int32), bs)
            y_true_transformed = transformer(y_true)
            y_true_transformed_noised = y_true_transformed + np.random.randn(*(y_true_transformed.shape)) / 20.0
            y_true_transformed_noised = np.clip(y_true_transformed_noised, 0.0, 1.0)
            y_true_transformed_back = transformer.transform_back(y_true_transformed_noised)
            assert np.alltrue(y_true_transformed_back == y_true)
