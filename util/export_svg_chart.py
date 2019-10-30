import os
import json
import pygal


def export_train_val_svg(title, data_type, x_data, train_value, validation_value, svg_path):
    line = pygal.Line(title=title,
                      x_label_rotation=20,
                      show_minor_x_labels=False,
                      dots_size=2)

    line.x_labels = x_data
    line.x_labels_major = x_data[::10]

    line.x_title = 'Epochs'
    line.y_title = data_type

    line.add('Train', train_value, formatter=lambda x: "%.4f" % x)
    line.add('Validation', validation_value, formatter=lambda x: "%.4f" % x)

    line.legend_at_bottom = True
    line.render_to_file(svg_path)


def export_dataset_stratified_hist(train_dir, validation_dir, title, output_path):
    ds_stat = {}
    for root, dirnames, filenames in os.walk(train_dir):
        for file in filenames:
            label = os.path.split(root)[-1]
            if label not in ds_stat.keys():
                ds_stat[label] = {'train': 0}
            else:
                ds_stat[label]['train'] += 1

    for root, dirnames, filenames in os.walk(validation_dir):
        for file in filenames:
            label = os.path.split(root)[-1]
            if label not in ds_stat.keys() or 'validation' not in ds_stat[label].keys():
                ds_stat[label]['validation'] = 0
            else:
                ds_stat[label]['validation'] += 1

    chart = pygal.StackedBar(print_values=True)
    x_title = sorted([k for k in ds_stat.keys()])

    chart.title = title
    chart.x_title = 'Label'
    chart.y_title = 'Number'
    chart.x_labels = x_title
    chart.legend_at_bottom = True

    chart.add('train', [ds_stat[i]['train'] for i in x_title])
    chart.add('validation', [ds_stat[i]['validation'] for i in x_title])

    chart.render_to_file(output_path)


if __name__ == '__main__':
    # data structure: [[time1, epoch1, acc1], [time2, epoch2, acc2], ...]
    # train_acc_data_file = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/graph/KFold.5.SET0.BATCH64.EPOCHS100.acc.json'
    # val_acc_data_file = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/graph/KFold.5.SET0.BATCH64.EPOCHS100.val_acc.json'
    # train_loss_data_file = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/graph/KFold.5.SET0.BATCH64.EPOCHS100.loss.json'
    # val_loss_data_file = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/graph/KFold.5.SET0.BATCH64.EPOCHS100.val_loss.json'
    #
    # with open(train_acc_data_file) as f:
    #     train_acc_data = json.load(f)
    # with open(val_acc_data_file) as f:
    #     val_acc_data = json.load(f)
    # with open(train_loss_data_file) as f:
    #     train_loss_data = json.load(f)
    # with open(val_loss_data_file) as f:
    #     val_loss_data = json.load(f)
    #
    # export_train_val_svg('Train/Validation loss of KFold.5.SET0.B64.E100',
    #                      'Loss',
    #                      [i[1] for i in train_loss_data],
    #                      [i[2] for i in train_loss_data],
    #                      [i[2] for i in val_loss_data],
    #                      '/Users/xie/Desktop/KFold.5.SET0.B64.E100.train_val_loss.svg')

    train_dir = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/data/0/train'
    validation_dir = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/data/0/validation'
    title = 'Stratified dataset in KFold.5'
    output_path = '/Users/xie/Desktop/KFold.5.stratified.svg'
    export_dataset_stratified_hist(train_dir, validation_dir, title, output_path)