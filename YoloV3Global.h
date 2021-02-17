#ifndef YOLOV3_GLOBAL_H
#define YOLOV3_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(YOLOV3_LIBRARY)
#  define YOLOV3SHARED_EXPORT Q_DECL_EXPORT
#else
#  define YOLOV3SHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // YOLOV3_GLOBAL_H
