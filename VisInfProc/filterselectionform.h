#ifndef FILTERSELECTIONFORM_H
#define FILTERSELECTIONFORM_H

#include <QDialog>

#include "filtersettings.h"

namespace Ui
{
class FilterSelectionForm;
}

class FilterSelectionForm : public QDialog
{
    Q_OBJECT

public:
    explicit FilterSelectionForm(QWidget *parent = 0);
    ~FilterSelectionForm();

    void show(QVector<float> allOrientations,
              QVector<FilterSettings> allSettings,
              QVector<int> activeOrientationIndices,
              QVector<int> activeSettingIndices);


signals:
    void activeFiltersChanged(QVector<int> activeOrientationIndices,
                              QVector<int> activeSettingIndices);

public slots:
    void onSubmit();

private:
    void clearLayout(QLayout *layout);

    Ui::FilterSelectionForm *ui;


};

#endif // FILTERSELECTIONFORM_H
