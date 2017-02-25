#ifndef MYQGRAPHICSIMAGE_H
#define MYQGRAPHICSIMAGE_H

#include <QGraphicsItem>
#include <QPainter>

class MyQGraphicsImage: public QGraphicsItem
{
public:
    MyQGraphicsImage():QGraphicsItem (NULL)
    {

    }

    void setImage ( const QImage & img)
    {
        this->img = img;
        update();
    }

    QRectF boundingRect() const
    {
        return QRectF(0, 0, img.width(), img.height());
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
    {
        painter->drawImage(0, 0, img);
    }

private:
    QImage img;
};

#endif // MYQGRAPHICSIMAGE_H
