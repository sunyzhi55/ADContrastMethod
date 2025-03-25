import sys
import torch
import torch.utils.data
from tqdm import tqdm

def run_main_1(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
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
            mri_images = mri_images.cuda(non_blocking=True)
            pet_image = pet_image.cuda(non_blocking=True)
            cli_tab = cli_tab.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad()
            mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
            # print(f'feature before loss{mri_feature.shape}')
            loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (mri_images, pet_image, cli_tab, label) in enumerate(test_bar):
                mri_images = mri_images.cuda(non_blocking=True)
                pet_image = pet_image.cuda(non_blocking=True)
                cli_tab = cli_tab.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                _, _, _, outputs = model.forward(mri_images, pet_image, cli_tab)
                prob_positive = outputs[:, 1]
                _, predictions = torch.max(outputs, dim=1)
                observer.update(predictions, prob_positive, label)
        if observer.excute(epoch, model=model):
            print("Early stopping")
            break
    observer.finish()

def run_main(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for ii, (mri_images, pet_image, cli_tab, label) in enumerate(train_bar):
            optimizer.zero_grad()
            mri_images = mri_images.cuda(non_blocking=True)
            pet_image = pet_image.cuda(non_blocking=True)
            cli_tab = cli_tab.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model.forward(mri_images, pet_image, cli_tab)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (mri_images, pet_image, cli_tab, label) in enumerate(test_bar):
                mri_images = mri_images.cuda(non_blocking=True)
                pet_image = pet_image.cuda(non_blocking=True)
                cli_tab = cli_tab.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                prob = model.forward(mri_images, pet_image, cli_tab)
                # _, predictions = torch.max(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                prob_positive = prob[:, 1]
                # predictions = (prob > 0.5).float()
                observer.update(predictions, prob_positive, label)

        if observer.excute(epoch, model=model):
            print("Early stopping")
            break
    observer.finish()

def run_main_for_IMF(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
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
            mri_images = mri_images.cuda(non_blocking=True)
            pet_image = pet_image.cuda(non_blocking=True)
            cli_tab = cli_tab.cuda(non_blocking=True)
            label_2d = label_2d.cuda(non_blocking=True)
            optimizer.zero_grad()
            # mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
            outputs = model.forward(mri_images, pet_image, cli_tab)
            # print(f'feature before loss{mri_feature.shape}')
            # loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
            loss = criterion(outputs, label_2d)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (mri_images, pet_image, cli_tab, label, label_2d) in enumerate(test_bar):
                mri_images = mri_images.cuda(non_blocking=True)
                pet_image = pet_image.cuda(non_blocking=True)
                cli_tab = cli_tab.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                # label_2d = label_2d.cuda(non_blocking=True)
                outputs = model.forward(mri_images, pet_image, cli_tab)
                prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
                _, predictions = torch.max(prob, dim=1)
                prob_positive = prob[:, 1]
                # predictions = (prob > 0.5).float()
                observer.update(predictions, prob_positive, label)

        if observer.excute(epoch, model=model):
            print("Early stopping")
            break
    observer.finish()


def run_main_for_MDL(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = lr_scheduler.get_last_lr()[0]

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}, LR {current_lr:.6f}", unit="batch", file=sys.stdout)

        for ii, (mri_images, pet_image, label) in enumerate(train_bar):
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(pet_image).any():
                print("train: NaN detected in input pet_image")
            mri_images = mri_images.cuda(non_blocking=True)
            pet_image = pet_image.cuda(non_blocking=True)
            input_data = torch.concat([mri_images, pet_image], dim=1)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad()
            # mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
            outputs, roi_out = model(input_data)
            # print(f'feature before loss{mri_feature.shape}')
            # loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            # test_bar = tqdm(test_loader, file=sys.stdout)
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch}", unit="batch", file=sys.stdout)
            for i, (mri_images, pet_image, label) in enumerate(test_bar):
                mri_images = mri_images.cuda(non_blocking=True)
                pet_image = pet_image.cuda(non_blocking=True)
                input_data = torch.concat([mri_images, pet_image], dim=1)
                label = label.cuda(non_blocking=True)
                outputs, roi_out = model(input_data)
                # print("outputs", outputs)
                prob = torch.softmax(outputs, dim=1)
                # print("prob", prob)
                _, predictions = torch.max(prob, dim=1)
                print("predictions", predictions)
                print("label", label)
                prob_positive = prob[:, 1]
                # predictions = (prob > 0.5).float()
                observer.update(predictions, prob_positive, label)
        if observer.excute(epoch, model=model):
            print("Early stopping")
            break
    observer.finish()

