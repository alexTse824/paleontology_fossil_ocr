import os
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
            if not os.path.splitext(file)[-1] == '.jpg':
                continue
            label = os.path.split(root)[-1]
            if label not in ds_stat.keys():
                ds_stat[label] = {'train': 0}
            else:
                ds_stat[label]['train'] += 1

    for root, dirnames, filenames in os.walk(validation_dir):
        for file in filenames:
            if not os.path.splitext(file)[-1] == '.jpg':
                continue
            label = os.path.split(root)[-1]
            if label not in ds_stat.keys() or 'validation' not in ds_stat[label].keys():
                ds_stat[label]['validation'] = 0
            else:
                ds_stat[label]['validation'] += 1

    print(ds_stat)
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
