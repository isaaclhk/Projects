# Standard library imports
import json
import logging
import os
import warnings
from copy import deepcopy

# Third-party imports
import hydra
from hydra.utils import instantiate, get_original_cwd
import mlflow
from mlflow.models.signature import infer_signature
import torch

# Local application/library-specific imports
import general_utils
from data_loader import PneumoniaDataset, get_dataloader, load_data, split_dataset, transform
from train_utils import compute_class_weights, EarlyStopping
from train_utils import train_step, val_step, set_seed, replace_classifier
from evaluate import evaluate

# Suppress warnings
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="../conf", config_name="main.yaml")
def main(
    args
) -> None:
    """
    Train a model and evaluate its performance on a validation set.

    This function performs training over a specified number of epochs, logging the 
    training and validation loss, as well as a specified evaluation metric 
    at each epoch. Optionally, this function also evalutes the model's performance 
    on a test set.

    Parameters:
        args: 
            Configuration arguments provided by Hydra.
    
    Note:
        After training, the model checkpoint is saved in the directory
        specified in the configuration file under 'model_checkpoint_dir_path'.
    """
    # Set seed for reproducibility
    set_seed(args["seed"])

    # setup logger
    logger = logging.getLogger(__name__)
    logger.info('Setting up logging configuration.')
    general_utils.setup_logging(
        logging_config_path=os.path.join(
            get_original_cwd(), "conf", "logging.yaml"
        ),
        log_dir=args.get("log_dir", None)
        )

    # init mlflow
    mlflow_init_status, mlflow_run, step_offset = general_utils.mlflow_init(
        tracking_uri = args.mlflow_tracking_uri,
        exp_name = args.mlflow_exp_name,
        run_name = args.mlflow_run_name,
        setup_mlflow = args.setup_mlflow,
        autolog = args.mlflow_autolog,
        resume = args.resume
    )

    ## Log hyperparameters used to train the model
    general_utils.mlflow_log(
        mlflow_init_status, 
        'log_params', 
        params = {
            'batch_size': args.batch_size,
            'lr': args.optimizer.lr,
            'epochs': args.epochs,
            'seed': args.seed
            }
        )

    # set device
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        logger.debug("device is set as cuda.")
    elif not args["no_mps"] and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("device is set as mps.")
    else:
        device = torch.device("cpu")
        logger.debug("device is set as cpu.")
    
    # define model
    model = instantiate(args['model'])

    # check whether classification is binary
    if args['metric']['num_classes'] < 2:
        logger.error("Invalid num_classes value.")
        raise ValueError(
            "num_classes must be >= 2."
        )
    else:
        is_binary = args['metric']['num_classes'] == 2

    # check whether training is required
    if args['epochs'] < 0:
        logger.error("Invalid epochs value.")
        raise ValueError(
            "epochs must be at least 0."
        )
    else:
        skip_training = args['epochs'] == 0
    if skip_training:
        logger.info("Epochs set to 0. Skipping model training..")

    # replace the last layer of the classifier
    model = replace_classifier(model, args['metric']['num_classes'])

    # freeze all layers except classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    if args['fine_tune']:
        # Unfreeze 2 more blocks in the features
        for layer in list(model.features.children())[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    # optimize model
    model.to(device)
    # model = torch.compile(model, 
    #                       backend='cudagraphs'
    #             )

    # define optimizer, metric, loss_fn (val and test)
    optimizer = instantiate(
        args['optimizer'], params=filter(lambda p: p.requires_grad, model.parameters())
        )
    metric = instantiate(args['metric']).to(device)
    
    valtest_loss_fn = instantiate(args['loss_fn'])

    # load state dictionary from model checkpoint path if resume training
    if args['load_checkpoint']:
        load_checkpoint_path = os.path.join(
            get_original_cwd(), args['load_checkpoint']
            )
        logger.info("Loading checkpoint from %s", load_checkpoint_path)
        try:
            model.load_state_dict(
                torch.load(
                    load_checkpoint_path, 
                    weights_only=True
                    )
                    )
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)
            
    # define data path and download data
    data_path = os.path.join(get_original_cwd(), args['data_path'])
    load_data(data_path)

    if not skip_training:
        # prepare metrics
        train_metric = deepcopy(metric)
        val_metric = deepcopy(metric)

        train_metric.to(device)
        val_metric.to(device)

        # Construct the full path relative to the original working directory
        checkpoint_dir = os.path.join(get_original_cwd(), args['model_checkpoint_dir_path'])
        # Create the directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.debug(
            'Created directory to save model checkpoints at %s.', checkpoint_dir
            )
        model_checkpoint_path = os.path.join(checkpoint_dir, args['model_checkpoint_name'])

        # datasets
        trainval_dataset = PneumoniaDataset(
            os.path.join(data_path, 'trainval'), 
            transform=transform(augment=True)
            )
        train_dataset, val_dataset = split_dataset(
            trainval_dataset, args['val_split'], args['seed'])
        
        # dataloaders
        train_dataloader = get_dataloader(
            train_dataset,
            shuffle=True,
            batch_size=args['batch_size'],
            seed=args['seed']
        )
        val_dataloader = get_dataloader(
            val_dataset,
            shuffle=False,
            batch_size=args['batch_size']
        )
        
        if args['balance_loss_weights']:
            try:
                weights = compute_class_weights(
                    train_dataset, args['metric']['num_classes']
                    )
                weights = weights.to(device)
                train_loss_fn = instantiate(args['loss_fn'], weight=weights)
                logger.debug(f"train loss function weights set to: {weights}")
            except Exception as e:
                logger.warning(f"weights not set for train loss function: {e}")
        else:
            train_loss_fn = instantiate(args['loss_fn'])
            logger.debug(f"No weights set for train loss function")

        # get input example for mlflow logging
        for batch in train_dataloader:
            input_example_, _ = batch
            break
        # Move input to the same device as the model
        input_example_ = input_example_.to(device)

        # forward pass and convert to numpy for mlflow
        output_example = model(input_example_).detach().cpu().numpy()
        input_example = input_example_.cpu().numpy()
        signature = infer_signature(input_example, output_example)

        if args["early_stopping"]:
            stopper = EarlyStopping(patience=args["patience"], path=model_checkpoint_path)

        for epoch in range(step_offset + 1, args['epochs'] + step_offset + 1):
            batch_loss, train_metric_score = train_step(
                model, train_metric, is_binary, train_dataloader, device, optimizer, train_loss_fn
                )

            val_batch_loss, val_metric_score = val_step(
                model, val_metric, is_binary, val_dataloader, device, valtest_loss_fn
            )

            # Print training and validation metrics
            logger.info(f"Epoch {epoch}/{args['epochs']} | "
                f"Train Loss: {batch_loss:.4f}, Train metric: {train_metric_score:.4f} | "
                f"Val Loss: {val_batch_loss:.4f}, Val metric: {val_metric_score:.4f}")

            # log metrics
            general_utils.mlflow_log(
                mlflow_init_status,
                "log_metrics",
                metrics = {
                    "Train Loss": batch_loss,
                    "Val Loss": val_batch_loss,
                    "Train metric": train_metric_score,
                    "Val metric": val_metric_score
                },
                step=epoch
            )

            # check early stopping
            if args['early_stopping']:
                stopper(val_metric_score, model) # save checkpoint if > best
                if stopper.best_score == val_metric_score:
                    # log checkpoint
                    general_utils.mlflow_log(
                    mlflow_init_status,
                    'log_artifact',
                    local_path = model_checkpoint_path
                    )

                if stopper.early_stop:
                    logger.info('Early stopping triggered. Model training completed.')
                    # load the best model from checkpoint
                    model.load_state_dict(
                        torch.load(model_checkpoint_path)
                        )
                    best_model = model.to(device)
                    # log model
                    general_utils.mlflow_pytorch_call(
                        mlflow_init_status,
                        "log_model",
                        pytorch_model=best_model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        pip_requirements="requirements.txt"
                    )
                    break 

            else: # if no early stopping, save model at checkpoint interval or last epoch
                if epoch % args["model_checkpoint_interval"] == 0:
                    try:
                        torch.save(model.state_dict(), model_checkpoint_path)
                        # log checkpoint
                        general_utils.mlflow_log(
                            mlflow_init_status,
                            'log_artifact',
                            local_path = model_checkpoint_path
                            )
                    except Exception as e:
                        logger.warning('Failed to save model checkpoint: %s', e)

                if epoch == args['epochs']:
                    logger.info("Maximum epochs reached. Model training completed.")
                    try:
                        torch.save(model.state_dict(), model_checkpoint_path)
                        # log checkpoint
                        general_utils.mlflow_log(
                            mlflow_init_status,
                            'log_artifact',
                            local_path = model_checkpoint_path
                            )
                    except Exception as e:
                        logger.warning("Failed to save model checkpoint: %s", e)
                            
                    # log model
                    general_utils.mlflow_pytorch_call(
                        mlflow_init_status,
                        "log_model",
                        pytorch_model=model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        pip_requirements="requirements.txt"
                    )

    if args['evaluate']:
        # load test data
        test_dataset = PneumoniaDataset(
            os.path.join(data_path, 'test'),
            transform=transform(augment=False)
        )
        test_dataloader = get_dataloader(
            test_dataset, 
            shuffle=False,
            batch_size=args['batch_size'])

        # construct dir to save evaluation artifacts
        evaluate_dir = os.path.join(get_original_cwd(), args['evaluate_dir_path'])
        os.makedirs(evaluate_dir, exist_ok=True)

        # prepare test metrics
        test_metric = deepcopy(metric)
        test_metric.to(device)

        # Evaluation
        logger.info("Performing Evaluation..")
        test_loss, test_score, preds_list, fig = evaluate(
            model, test_metric, is_binary, device, test_dataloader, valtest_loss_fn)

        # save predictions to json file
        preds_path = os.path.join(evaluate_dir, 'predictions.json')
        try:
            preds_list = [int(pred) for pred in preds_list] # json-serializable
            with open(preds_path, "w") as f:
                json.dump(preds_list, f)
            logger.info(f"Predictions saved to {preds_path}")
        except Exception as e:
            logger.error(f'Failed to save predictions to {preds_path}: {e}')
        
        # save confusion matrix figure
        cm_path = os.path.join(evaluate_dir, 'confusion_matrix.png')
        try:
            fig.savefig(cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix to {cm_path}: {e}")

        # log test metrics
        general_utils.mlflow_log(
            mlflow_init_status,
            "log_metrics",
            metrics = {
                "Test Loss": test_loss,
                "Test Score": test_score,
            })
        # log predictions
        general_utils.mlflow_log(
                mlflow_init_status,
                'log_artifact',
                local_path = preds_path
            )
        # log confusion matrix
        general_utils.mlflow_log(
                mlflow_init_status,
                'log_artifact',
                local_path = cm_path
            )
        
        ## Use MLflow to log artifact (model config in json)
        general_utils.mlflow_log(
            mlflow_init_status,
            'log_dict',
            dictionary = args,
            artifact_file = args.model_config_dir_path
        )

        ## Use MLflow to log artifacts (entire `logs`` directory)
        general_utils.mlflow_log(
            mlflow_init_status,
            'log_artifact',
            local_path = args.log_dir
        )

    if mlflow_init_status:
        ## Get artifact link
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: %s", artifact_uri)
        general_utils.mlflow_log(
            mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
        )
        logger.info(
            "Model run with MLflow run ID %s has completed.",
            mlflow_run.info.run_id,
        )
        mlflow.end_run()
    else:
        logger.info("Model run has ended.")

if __name__ == "__main__":
    main()