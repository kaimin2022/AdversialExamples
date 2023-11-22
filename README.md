# ACVAEGAN
XLGI: A StyleGAN-XL-Based Gradient Inversion Attack against Federated Learning

## Description
- Goal: Reconstructing federal learning users' privacy data by acquiring gradient information.
- Architecture:
![ACVAEGAN Architecture](https://github.com/kaimin2022/Gradient-Inversion/blob/main/GIXL.png))
- Dataset: CIFAR10、FFHQ、TanyImageNet
- Approaches: 
  1. Get privacy data gradient information
  2. Latent code is generated randomly and input into the StyleGAN-XL model to generate dummy data
  3. Obtain the gradient of the virtual picture and calculate the loss between the gradients to optimize Latent code
  

 
## References
[1] Boenisch, F., Dziedzic, A., Schuster, R., Shamsabadi, A.S., Shumailov, I., Papernot, N., 2021. When the curious abandon honesty: Federated
learning is not private. arXiv preprint arXiv:2112.02918 .

[2] Cao, D., Wei, K., Wu, Y., Zhang, J., Feng, B., Chen, J., 2023. Fepn: A robust feature purification network to defend against adversarial examples.
Computers & Security 134, 103427.

[3] Hatamizadeh, A., Yin, H., Roth, H.R., Li, W., Kautz, J., Xu, D., Molchanov, P., 2022. Gradvit: Gradient inversion of vision transformers, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), pp. 10021–10030.

[4] Jeon, J., Lee, K., Oh, S., Ok, J., et al., 2021. Gradient inversion with generative image prior. Advances in neural information processing systems ,
29898–29908.

[5] Zhao, B., Mopuri, K.R., Bilen, H., 2020. idlg: Improved deep leakage from gradients. arXiv preprint arXiv:2001.02610 .

[6] Zhu, J., Blaschko, M.B., 2021. R-{gap}: Recursive gradient attack on privacy, in: Proceedings of the International Conference on Learning
Representations.

[7] Zhu, L., Liu, Z., Han, S., 2019. Deep leakage from gradients. Advances in neural information processing systems 32
