#include "aspectratiopixmap.h"

AspectRatioPixmap::AspectRatioPixmap(QWidget *parent) :
    QGraphicsView(parent)
{
    this->setMinimumSize(1,1);
    scene = new QGraphicsScene();
    pix = new MyQGraphicsImage();
    scene->addItem(pix);
    setScene(scene);
    show();
}
AspectRatioPixmap::~AspectRatioPixmap()
{
    delete scene;
}

void AspectRatioPixmap::setImage ( const QImage & img)
{
    pix->setImage(img.mirrored());
    setSceneRect(pix->boundingRect());
    fitInView(scene->itemsBoundingRect(),Qt::KeepAspectRatio);
}

void AspectRatioPixmap::clear()
{
    pix->setImage(QImage());
}
