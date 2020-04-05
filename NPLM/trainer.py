import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *

def trainer(model, loss_func, optimizer, dataloader, device, num_epochs, check_points=None):
    if check_points is not None:
        load_check_points(check_points, model, optimizer, device, mode='train')
    else:
        losses = []
        for epo in range(num_epochs):
            total_loss = .0
            for i, v in enumerate(dataloader):
                contexts = v[0].to(device)
                target = v[1].to(device)
                
                model.zero_grad()

                probs = model(contexts)

                loss = loss_func(probs, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            losses.append(total_loss)
            print(epo, 'th epoch loss :', total_loss)
        
        save_check_points(check_points, model, num_epochs, total_loss)
    
    return model, losses