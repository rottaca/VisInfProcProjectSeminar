#include "filterselectionform.h"
#include "ui_filterselectionform.h"

#include <QCheckBox>
#include <QMessageBox>

#include "settings.h"

FilterSelectionForm::FilterSelectionForm(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FilterSelectionForm)
{
    ui->setupUi(this);
    setModal(true);

    connect(ui->b_submit,SIGNAL(clicked()),this,SLOT(onSubmit()));
    connect(ui->b_cancel,SIGNAL(clicked()),this,SLOT(close()));
}

FilterSelectionForm::~FilterSelectionForm()
{
    delete ui;
}

void FilterSelectionForm::show(QVector<float> allOrientations,
                               QVector<FilterSettings> allSettings,
                               QVector<int> activeOrientationIndices,
                               QVector<int> activeSettingIndices)
{
    clearLayout(ui->vLayout_orientations);
    clearLayout(ui->vLayout_speeds);

    int i = 0;
    Q_FOREACH(float o,allOrientations) {
        QCheckBox *cb = new QCheckBox(this);
        cb->setProperty("idx",i);
        cb->setText(QString("%1°").arg(RAD2DEG(o)));
        if(activeOrientationIndices.contains(i))
            cb->setChecked(true);
        ui->vLayout_orientations->addWidget(cb);
        i++;
    }
    i = 0;
    Q_FOREACH(FilterSettings fs,allSettings) {
        QCheckBox *cb = new QCheckBox(this);
        cb->setProperty("idx",i);
        cb->setText(QString("%1 px/sec").arg(fs.speed_px_per_sec));
        if(activeSettingIndices.contains(i))
            cb->setChecked(true);
        ui->vLayout_speeds->addWidget(cb);
        i++;
    }
    QDialog::show();
}

void FilterSelectionForm::clearLayout(QLayout *layout)
{
    QLayoutItem *item;
    while((item = layout->takeAt(0))) {
        if (item->layout()) {
            clearLayout(item->layout());
            delete item->layout();
        }
        if (item->widget()) {
            delete item->widget();
        }
        delete item;
    }
}
void FilterSelectionForm::onSubmit()
{
    QVector<int> activeOrientationIndices;
    QVector<int> activeSettingIndices;

    int i = 0;
    QLayoutItem *item;
    while((item = ui->vLayout_orientations->itemAt(i))) {
        QCheckBox* cb = (QCheckBox*)item->widget();
        if(cb->isChecked())
            activeOrientationIndices.append(cb->property("idx").toInt());
        i++;
    }
    i = 0;
    while((item = ui->vLayout_speeds->itemAt(i))) {
        QCheckBox* cb = (QCheckBox*)item->widget();
        if(cb->isChecked())
            activeSettingIndices.append(cb->property("idx").toInt());
        i++;
    }
    if(activeOrientationIndices.size() == 0 || activeSettingIndices.size() == 0) {
        QMessageBox::critical(this,"Invalid selection",
                              "At least one speed and one orientation have to be selected!");
    } else {
        emit activeFiltersChanged(activeOrientationIndices,activeSettingIndices);
        close();
    }
}
