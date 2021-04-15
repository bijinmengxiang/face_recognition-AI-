This folder have the network structure.
Fisrtly this network structure was inspired from mnist_net.
I changed some of the kernel_size and channels.
Beyond my imagination, it's worked. 
Original size of image was 84*84.

(First Version)
Because of less dataset,I use the function of RandomHorizontalFlip that can make image more diversity.
And I trained the dataset 50 times.
Then I try to use the model in my testset, the Accuracy was 85.22%.
The learning rate and momentum are 0.015 and 0.5.
I have already tried other parameter including reduce learning rate and increase Train Epoch,
the effect is almost as same as first version.
(when Train Epoch are 80 times and learning rate is 0.1 the Accuracy was increased to 86.2%.)

---------------------------------------------------------------------------------------
