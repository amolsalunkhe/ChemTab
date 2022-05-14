# Verified to work! 5/14/22 (in regards to relative tolerance, see the assert)
# :return: rescaling_layer, (m,b) <-- includes params for manual scaling!
def fit_rescaling_layer(output_scaler, layer_name='rescaling', n_samples=1000):
    import sklearn, tensorflow as tf, tensorflow.keras as keras, numpy as np # metrics ... for sanity checks
    def R2(yt,yp): return tf.reduce_mean(1-tf.reduce_mean((yp-yt)**2, axis=0)/(tf.math.reduce_std(yt,axis=0)**2))
    def rel_err(yt, yp): return tf.reduce_mean(tf.abs((yp-yt)/yt))
    
    n_input_features = output_scaler.n_features_in_
    data_scale = 100
    dummy_input_data = (np.random.random(size=(n_samples,n_input_features))-0.5)*data_scale
    inverted_data = output_scaler.inverse_transform(dummy_input_data)
    def fit_rescaling_lms():
        linear_models = []
        for i in range(n_input_features):
            lm = sklearn.linear_model.LinearRegression()
            X_data = dummy_input_data[:,i].reshape(-1,1)
            Y_data = inverted_data[:,i].reshape(-1,1)
            lm.fit(X_data, Y_data)
            assert lm.score(X_data, Y_data)==1.0 # assert R^2==1 (should be perfect fit)
            linear_models.append(lm)
        return linear_models
    lms = fit_rescaling_lms()
    m = np.array([lm.coef_ for lm in lms]).squeeze()
    b = np.array([lm.intercept_ for lm in lms]).squeeze()
    rescaling_layer = keras.layers.Rescaling(m, b, name=layer_name) # y=mx+b
    rescale_inverted_data = rescaling_layer(dummy_input_data).numpy().astype('float64')
    print('MAE/data_scale for inversion layer:', np.mean(np.abs(rescale_inverted_data-inverted_data))/data_scale)
    print('R^2 for inversion layer:', R2(inverted_data, rescale_inverted_data).numpy())
    print('Rel-error for inversion layer:', rel_err(inverted_data, rescale_inverted_data).numpy())
    assert np.allclose(rescale_inverted_data, inverted_data)
    
    return rescaling_layer, (m, b)

#rescaling_layer, (m,b) = fit_rescaling_layer(dm.outputScaler)
