# Style Transfer
Style Transfer with keras and tensorflow

Based on:
https://github.com/simulacre7/tensorflow-IPythonNotebook/tree/master/neural-style


# neural style

An implementation of [neural style][paper] written in TensorFlow with IPythonNotebook.

## Examples

These were the input images used :

![input-content](images/1-content.jpg)

![input-style](images/1-style.jpg)

This is the output produced by the algorithm:
![output](images/output_1-content.jpg)

## Details

TensorFlow doesn't support [L-BFGS][l-bfgs] which is the original authors used.
So we use [Adam][adam]. This may require a little bit more hyperparameter tuning to get nice results.

you can get Pre-trained VGG network by

`wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat`


## reference
[A Neural Algorithm of Artistic Style (Leon A. Gatys, et al.)][paper]

[Exploring the Neural Algorithm of Artistic Style (Yaroslav Nikulin, et al)][paper2]

[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[paper2]: http://arxiv.org/pdf/1602.07188v1.pdf
[style]: http://www.ebsqart.com/Art-Galleries/Contemporary-Cubism/43/Cubist-9/204218/
[rain]: https://afremov.com/RAIN-PRINCESS-Palette-knife-Oil-Painting-on-Canvas-by-Leonid-Afremov-Size-30-x30.html
[UNIST]: http://www.studyinkorea.go.kr/en/sub/college_info/college_info.do?ei_code=562240
[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
[adam]: http://arxiv.org/abs/1412.6980

