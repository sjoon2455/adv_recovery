attacks = [
    # L1
    fa.EADAttack(),
    fa.L1BrendelBethgeAttack(),

    # L_2
    fa.L2ContrastReductionAttack(),
    fa.L2ProjectedGradientDescentAttack(),
    fa.L2AdditiveGaussianNoiseAttack(),
    fa.L2AdditiveUniformNoiseAttack(),
    fa.L2ClippingAwareAdditiveGaussianNoiseAttack(),
    fa.L2ClippingAwareAdditiveUniformNoiseAttack(),
    fa.L2FastGradientAttack(),
    fa.L2RepeatedAdditiveGaussianNoiseAttack(),
    fa.L2RepeatedAdditiveUniformNoiseAttack(),
    fa.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(),
    fa.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(),
    fa.L2DeepFoolAttack(),
    fa.L2BrendelBethgeAttack(),
    fa.L2PGD(),
    fa.L2CarliniWagnerAttack(),
    fa.DDNAttack(),

    # L_inf
    fa.LinfProjectedGradientDescentAttack(),
    fa.LinfBasicIterativeAttack(),
    fa.LinfFastGradientAttack(),
    fa.LinfAdditiveUniformNoiseAttack(),
    fa.LinfRepeatedAdditiveUniformNoiseAttack(),
    fa.LinfDeepFoolAttack(),
    fa.LinfinityBrendelBethgeAttack(),
    fa.LinfPGD(),

    fa.InversionAttack(),  # invert pixel
    fa.BinarySearchContrastReductionAttack(),
    fa.LinearSearchContrastReductionAttack(),
    fa.GaussianBlurAttack(),
    fa.SaltAndPepperNoiseAttack(),  # until misclassified
    fa.LinearSearchBlendedUniformNoiseAttack(),
    fa.BoundaryAttack(),  # reduce perturbation while staying adversarial
    # fa.BinarizationRefinementAttack(), #for models that preprocess inputs by binarizing the inputs
    # fa.DatasetAttack(), # needs secondary dataset to be drawn from

    # gradient
    fa.L0BrendelBethgeAttack(),
    fa.NewtonFoolAttack(),
    # fa.VirtualAdversarialAttack(),
]


'''
    attacks_2 = [
        # L1
        fa.EADAttack(),
        fa.L1BrendelBethgeAttack(),

        # L_2
        fa.L2ContrastReductionAttack(),
        fa.L2ProjectedGradientDescentAttack(),
        fa.L2AdditiveGaussianNoiseAttack(),
        fa.L2AdditiveUniformNoiseAttack(),
        # fa.L2ClippingAwareAdditiveGaussianNoiseAttack(),
        # fa.L2ClippingAwareAdditiveUniformNoiseAttack(),
        fa.L2FastGradientAttack(),
        fa.L2RepeatedAdditiveGaussianNoiseAttack(),
        fa.L2RepeatedAdditiveUniformNoiseAttack(),
        # fa.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(),
        # fa.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(),
        fa.L2DeepFoolAttack(),
        fa.L2BrendelBethgeAttack(),
        fa.L2PGD(),
        fa.L2CarliniWagnerAttack(),
        fa.DDNAttack(),

        # L_inf
        fa.LinfProjectedGradientDescentAttack(),
        fa.LinfBasicIterativeAttack(),
        fa.LinfFastGradientAttack(),
        fa.LinfAdditiveUniformNoiseAttack(),
        fa.LinfRepeatedAdditiveUniformNoiseAttack(),
        fa.LinfDeepFoolAttack(),
        fa.LinfinityBrendelBethgeAttack(),
        fa.LinfPGD(),

        fa.InversionAttack(),  # invert pixel
        fa.BinarySearchContrastReductionAttack(),
        fa.LinearSearchContrastReductionAttack(),
        fa.GaussianBlurAttack(),
        fa.SaltAndPepperNoiseAttack(),  # until misclassified
        fa.LinearSearchBlendedUniformNoiseAttack(),
        fa.BoundaryAttack(),  # reduce perturbation while staying adversarial

        # gradient
        fa.L0BrendelBethgeAttack(),
        fa.NewtonFoolAttack(),
    ]
    '''
