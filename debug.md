**RuntimeError: Invalid device string: '0'** : 
baseline_main.py中第21行 *torch.cuda.set_device(args.gpu)*
改成 *torch.cuda.set_device(int(args.gpu))*

**FileNotFoundError: [Errno 2] No such file or directory: '/save/nn_mnist_mlp_1.png'** : 
baseline_main.py中第93行 *plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))*
改成 *plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))*
確認自己cmd路徑來決定是 '../save' or './save'

**AttributeError: 'Namespace' object has no attribute 'gpu_id'**
federated_main.py中第32、33行*if args.gpu_id:    torch.cuda.set_device(args.gpu_id)*
改成 *if args.gpu:   torch.cuda.set_device(int(args.gpu))*