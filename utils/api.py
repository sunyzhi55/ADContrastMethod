import sys
import torch
import torch.utils.data
from tqdm import tqdm

def run_main_1(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        for ii, (mri_images, pet_image, cli_tab, label) in enumerate(train_bar):
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(pet_image).any():
                print("train: NaN detected in input pet_image")
            mri_images = mri_images.to(device)
            pet_image = pet_image.to(device)
            cli_tab = cli_tab.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
            # print(f'feature before loss{mri_feature.shape}')
            loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
            _, predictions = torch.max(outputs, dim=1)
            prob_positive = outputs[:, 1]
            # predictions = (prob > 0.5).float()
            observer.train_update(loss, predictions, prob_positive, label)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)
            for i, (mri_images, pet_image, cli_tab, label) in enumerate(test_bar):
                mri_images = mri_images.to(device)
                pet_image = pet_image.to(device)
                cli_tab = cli_tab.to(device)
                label = label.to(device)
                mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
                loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
                _, predictions = torch.max(outputs, dim=1)
                prob_positive = outputs[:, 1]
                # predictions = (prob > 0.5).float()
                observer.eval_update(loss, predictions, prob_positive, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset),len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)

def run_main(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for ii, (mri_images, pet_image, cli_tab, label) in enumerate(train_bar):
            optimizer.zero_grad()
            mri_images = mri_images.to(device)
            pet_image = pet_image.to(device)
            cli_tab = cli_tab.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            prob = model.forward(mri_images, pet_image, cli_tab)
            loss = criterion(prob, label)
            _, predictions = torch.max(prob, dim=1)
            prob_positive = prob[:, 1]
            loss.backward()
            optimizer.step()
            observer.train_update(loss, predictions, prob_positive, label)
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (mri_images, pet_image, cli_tab, label) in enumerate(test_bar):
                mri_images = mri_images.to(device)
                pet_image = pet_image.to(device)
                cli_tab = cli_tab.to(device)
                label = label.to(device)
                prob = model.forward(mri_images, pet_image, cli_tab)
                loss = criterion(prob, label)
                _, predictions = torch.max(prob, dim=1)
                prob_positive = prob[:, 1]
                # predictions = (prob > 0.5).float()
                observer.eval_update(loss, predictions, prob_positive, label)
        if observer.execute(epoch, model=model):
            print("Early stopping")
            break
    observer.finish(fold)

def run_main_for_IMF(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for ii, (mri_images, pet_image, cli_tab, label, label_2d) in enumerate(train_bar):
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(pet_image).any():
                print("train: NaN detected in input pet_image")
            mri_images = mri_images.to(device)
            pet_image = pet_image.to(device)
            cli_tab = cli_tab.to(device)
            label = label.to(device)
            label_2d = label_2d.to(device)
            optimizer.zero_grad()
            # mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
            outputs = model.forward(mri_images, pet_image, cli_tab)
            # print(f'feature before loss{mri_feature.shape}')
            # loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
            loss = criterion(outputs, label_2d)
            prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
            _, predictions = torch.max(prob, dim=1)
            prob_positive = prob[:, 1]
            loss.backward()
            optimizer.step()
            observer.train_update(loss, predictions, prob_positive, label)
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)
            for i, (mri_images, pet_image, cli_tab, label, label_2d) in enumerate(test_bar):
                mri_images = mri_images.to(device)
                pet_image = pet_image.to(device)
                cli_tab = cli_tab.to(device)
                label = label.to(device)
                label_2d = label_2d.to(device)
                outputs = model.forward(mri_images, pet_image, cli_tab)
                loss = criterion(outputs, label_2d)
                prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
                _, predictions = torch.max(prob, dim=1)
                prob_positive = prob[:, 1]
                # predictions = (prob > 0.5).float()
                observer.eval_update(loss, predictions, prob_positive, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset),len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_MDL(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = lr_scheduler.get_last_lr()[0]

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}, LR {current_lr:.6f}", unit="batch", file=sys.stdout)

        for ii, (gm_img_torch, wm_img_torch, pet_img_torch, label) in enumerate(train_bar):
            if torch.isnan(gm_img_torch).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(wm_img_torch).any():
                print("train: NaN detected in input pet_image")
            if torch.isnan(pet_img_torch).any():
                print("train: NaN detected in input pet_image")
            gm_img_torch = gm_img_torch.to(device)
            wm_img_torch = wm_img_torch.to(device)
            pet_img_torch = pet_img_torch.to(device)
            input_data = torch.concat([gm_img_torch, wm_img_torch, pet_img_torch], dim=1)
            label = label.to(device)
            optimizer.zero_grad()
            # mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
            outputs, roi_out = model(input_data)
            # print(f'feature before loss{mri_feature.shape}')
            # loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
            loss = criterion(outputs, label)
            prob = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(prob, dim=1)
            prob_positive = prob[:, 1]
            # predictions = (prob > 0.5).float()
            observer.train_update(loss, predictions, prob_positive, label)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            # test_bar = tqdm(test_loader, file=sys.stdout)
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch}", unit="batch", file=sys.stdout)
            for i, (gm_img_torch, wm_img_torch, pet_img_torch, label) in enumerate(test_bar):
                gm_img_torch = gm_img_torch.to(device)
                wm_img_torch = wm_img_torch.to(device)
                pet_img_torch = pet_img_torch.to(device)
                input_data = torch.concat([gm_img_torch, wm_img_torch, pet_img_torch], dim=1)
                label = label.to(device)
                outputs, roi_out = model(input_data)
                loss = criterion(outputs, label)
                prob = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                prob_positive = prob[:, 1]
                # predictions = (prob > 0.5).float()
                observer.eval_update(loss, predictions, prob_positive, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset),len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_RLAD(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = lr_scheduler.get_last_lr()[0]

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}, LR {current_lr:.6f}", unit="batch", file=sys.stdout)

        for ii, (mri_images, label) in enumerate(train_bar):
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            mri_images = mri_images.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            _, outputs, _ = model(mri_images)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            # test_bar = tqdm(test_loader, file=sys.stdout)
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch}", unit="batch", file=sys.stdout)
            for i, (mri_images, label) in enumerate(test_bar):
                mri_images = mri_images.to(device)
                label = label.to(device)
                _, outputs, _ = model(mri_images)
                prob = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                # print("predictions", predictions)
                # print("label", label)
                prob_positive = prob[:, 1]
                # predictions = (prob > 0.5).float()
                observer.update(predictions, prob_positive, label)
        if observer.execute(epoch, model=model):
            print("Early stopping")
            break
    observer.finish(fold)

