def trainer(model, loss_func, optimizer, dataloader, device, num_epochs):

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
    
    return model, losses