#ifndef ASPECTRATIOPIXMAP_H
#define ASPECTRATIOPIXMAP_H

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QResizeEvent>

#include "myqgraphicsimage.h"

class AspectRatioPixmap : public QGraphicsView
{
    Q_OBJECT
public:
    explicit AspectRatioPixmap(QWidget *parent = 0);
    ~AspectRatioPixmap();


public slots:
    void setImage ( const QImage & );
    void clear();

private:
    MyQGraphicsImage* pix;
    QGraphicsScene* scene;
};

#endif // ASPECTRATIOPIXMAP_H
