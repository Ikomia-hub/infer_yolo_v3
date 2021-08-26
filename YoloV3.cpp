#include "YoloV3.h"
#include "Graphics/CGraphicsLayer.h"

//------------------------//
//----- CYoloV3Param -----//
//------------------------//
CYoloV3Param::CYoloV3Param() : COcvDnnProcessParam()
{
    m_framework = Framework::DARKNET;
    m_inputSize = 416;
    m_modelName = "YOLOv3";
    m_datasetName = "COCO";
    m_modelFolder = Utils::Plugin::getCppPath() + "/YoloV3/Model/";
}

void CYoloV3Param::setParamMap(const UMapString &paramMap)
{
    COcvDnnProcessParam::setParamMap(paramMap);
    m_confidence = std::stod(paramMap.at("confidence"));
    m_nmsThreshold = std::stod(paramMap.at("nmsThreshold"));
}

UMapString CYoloV3Param::getParamMap() const
{
    auto paramMap = COcvDnnProcessParam::getParamMap();
    paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
    paramMap.insert(std::make_pair("nmsThreshold", std::to_string(m_nmsThreshold)));
    return paramMap;
}

//-------------------//
//----- CYoloV3 -----//
//-------------------//
CYoloV3::CYoloV3() : COcvDnnProcess()
{
    m_pParam = std::make_shared<CYoloV3Param>();
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

CYoloV3::CYoloV3(const std::string &name, const std::shared_ptr<CYoloV3Param> &pParam): COcvDnnProcess(name)
{
    m_pParam = std::make_shared<CYoloV3Param>(*pParam);
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

size_t CYoloV3::getProgressSteps()
{
    return 3;
}

int CYoloV3::getNetworkInputSize() const
{
    auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
    return pParam->m_inputSize;
}

double CYoloV3::getNetworkInputScaleFactor() const
{
    return 1.0 / 255.0;
}

cv::Scalar CYoloV3::getNetworkInputMean() const
{
    return cv::Scalar();
}

void CYoloV3::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);

    if(pInput == nullptr || pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    CMat imgSrc;
    CMat imgOrigin = pInput->getImage();
    std::vector<cv::Mat> netOutputs;

    //Detection networks need color image as input
    if(imgOrigin.channels() < 3)
        cv::cvtColor(imgOrigin, imgSrc, cv::COLOR_GRAY2RGB);
    else
        imgSrc = imgOrigin;

    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn();
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            pParam->m_bUpdate = false;
        }

        int size = getNetworkInputSize();
        double scaleFactor = getNetworkInputScaleFactor();
        cv::Scalar mean = getNetworkInputMean();
        auto inputBlob = cv::dnn::blobFromImage(imgSrc, scaleFactor, cv::Size(size,size), mean, false, false);
        m_net.setInput(inputBlob);

        auto netOutNames = getOutputsNames();
        m_net.forward(netOutputs, netOutNames);
    }
    catch(cv::Exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    endTaskRun();

    if(m_classNames.empty())
        readClassNames();

    emit m_signalHandler->doProgress();
    manageOutput(netOutputs);
    emit m_signalHandler->doProgress();
}

void CYoloV3::manageOutput(const std::vector<cv::Mat> &dnnOutputs)
{
    forwardInputImage();

    auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    CMat imgSrc = pInput->getImage();

    const size_t nbClasses = m_classNames.size();
    std::vector<std::vector<cv::Rect2d>> boxes;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<int>> indices;
    boxes.resize(nbClasses);
    scores.resize(nbClasses);
    indices.resize(nbClasses);

    //Graphics output
    auto pGraphicsOutput = std::dynamic_pointer_cast<CGraphicsOutput>(getOutput(1));
    pGraphicsOutput->setNewLayer(getName());
    pGraphicsOutput->setImageIndex(0);

    //Measures output
    auto pMeasureOutput = std::dynamic_pointer_cast<CBlobMeasureIO>(getOutput(2));
    pMeasureOutput->clearData();

    const int probabilityIndex = 5;
    for(auto&& output : dnnOutputs)
    {
        const auto nbBoxes = output.rows;
        for(int i=0; i<nbBoxes; ++i)
        {
            float xCenter = output.at<float>(i, 0) * imgSrc.cols;
            float yCenter = output.at<float>(i, 1) * imgSrc.rows;
            float width = output.at<float>(i, 2) * imgSrc.cols;
            float height = output.at<float>(i, 3) * imgSrc.rows;
            float left = xCenter - width/2;
            float top = yCenter - height/2;
            cv::Rect2d r(left, top, width, height);

            for(size_t j=0; j<nbClasses; ++j)
            {
                float confidence = output.at<float>(i, (int)j + probabilityIndex);
                if (confidence > pParam->m_confidence)
                {
                    boxes[j].push_back(r);
                    scores[j].push_back(confidence);
                }
            }
        }
    }

    // Apply non-maximum suppression
    for(size_t i=0; i<nbClasses; ++i)
        cv::dnn::NMSBoxes(boxes[i], scores[i], pParam->m_confidence, pParam->m_nmsThreshold, indices[i]);

    for(size_t i=0; i<nbClasses; ++i)
    {
        for(size_t j=0; j<indices[i].size(); ++j)
        {
            //Create rectangle graphics of bbox
            const int index = indices[i][j];
            cv::Rect2d box = boxes[i][index];
            auto graphicsObj = pGraphicsOutput->addRectangle(box.x, box.y, box.width, box.height);

            //Retrieve class label
            float confidence = scores[i][index];
            std::string label = m_classNames[i] + " : " + std::to_string(confidence);
            pGraphicsOutput->addText(label, box.x + 5, box.y + 5);

            //Store values to be shown in results table
            std::vector<CObjectMeasure> results;
            results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidence, graphicsObj->getId(), m_classNames[i]));
            results.emplace_back(CObjectMeasure(CMeasure::Id::BBOX, {box.x, box.y, box.width, box.height}, graphicsObj->getId(), m_classNames[i]));
            pMeasureOutput->addObjectMeasures(results);
        }
    }
}

//-------------------------//
//----- CYoloV3Widget -----//
//-------------------------//
CYoloV3Widget::CYoloV3Widget(QWidget *parent): COcvWidgetDnnCore(parent)
{
    init();
}

CYoloV3Widget::CYoloV3Widget(WorkflowTaskParamPtr pParam, QWidget *parent): COcvWidgetDnnCore(pParam, parent)
{
    m_pParam = std::dynamic_pointer_cast<CYoloV3Param>(pParam);
    init();
}

void CYoloV3Widget::init()
{
    if(m_pParam == nullptr)
        m_pParam = std::make_shared<CYoloV3Param>();

    auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
    assert(pParam);

    m_pSpinInputSize = addSpin(tr("Input size"), pParam->m_inputSize, 32, 2048, 32);

    m_pComboModel = addCombo(tr("Model"));
    m_pComboModel->addItem("CSResNeXt50-panet-spp-optimal");
    m_pComboModel->addItem("YOLOv3-spp");
    m_pComboModel->addItem("YOLOv3");
    m_pComboModel->addItem("Tiny YOLOv3");
    m_pComboModel->setCurrentText(QString::fromStdString(pParam->m_modelName));

    m_pComboDataset = addCombo(tr("Trained on"));
    m_pComboDataset->addItem("COCO");
    m_pComboDataset->addItem("Custom");
    m_pComboDataset->setCurrentText(QString::fromStdString(pParam->m_datasetName));

    m_pBrowseConfig = addBrowseFile(tr("Configuration file"), QString::fromStdString(pParam->m_structureFile), "");
    m_pBrowseConfig->setEnabled(pParam->m_datasetName == "Custom");

    m_pBrowseWeights = addBrowseFile(tr("Weights file"), QString::fromStdString(pParam->m_modelFile), "");
    m_pBrowseWeights->setEnabled(pParam->m_datasetName == "Custom");

    m_pBrowseLabels = addBrowseFile(tr("Labels file"), QString::fromStdString(pParam->m_labelsFile), "");
    m_pBrowseLabels->setEnabled(pParam->m_datasetName == "Custom");

    auto pSpinConfidence = addDoubleSpin(tr("Confidence"), pParam->m_confidence, 0.0, 1.0, 0.1, 2);
    auto pSpinNmsThreshold = addDoubleSpin(tr("NMS threshold"), pParam->m_nmsThreshold, 0.0, 1.0, 0.1, 2);

    //Connections
    connect(m_pComboModel, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int index)
    {
        Q_UNUSED(index);
        auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
        assert(pParam);
        pParam->m_modelName = m_pComboModel->currentText().toStdString();
        pParam->m_bUpdate = true;
    });
    connect(m_pComboDataset, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int index)
    {
        Q_UNUSED(index);
        auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
        assert(pParam);
        pParam->m_datasetName = m_pComboDataset->currentText().toStdString();
        m_pBrowseConfig->setEnabled(pParam->m_datasetName == "Custom");
        m_pBrowseWeights->setEnabled(pParam->m_datasetName == "Custom");
        m_pBrowseLabels->setEnabled(pParam->m_datasetName == "Custom");
        pParam->m_bUpdate = true;
    });
    connect(pSpinConfidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
    {
        auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
        assert(pParam);
        pParam->m_confidence = val;
    });
    connect(pSpinNmsThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
    {
        auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
        assert(pParam);
        pParam->m_nmsThreshold = val;
    });
}

void CYoloV3Widget::onApply()
{
    auto pParam = std::dynamic_pointer_cast<CYoloV3Param>(m_pParam);
    assert(pParam);
    pParam->m_inputSize = m_pSpinInputSize->value();

    if(pParam->m_datasetName == "COCO")
    {
        pParam->m_labelsFile = pParam->m_modelFolder + "coco_names.txt";
        m_pBrowseLabels->setPath(QString::fromStdString(pParam->m_labelsFile));

        if(pParam->m_modelName == "YOLOv3")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "yolov3.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "yolov3.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
        else if(pParam->m_modelName == "Tiny YOLOv3")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "yolov3-tiny.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "yolov3-tiny.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
        else if(pParam->m_modelName == "YOLOv3-spp")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "yolov3-spp.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "yolov3-spp.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
        else if(pParam->m_modelName == "CSResNeXt50-panet-spp-optimal")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "csresnext50-panet-spp-original-optimal.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "csresnext50-panet-spp-original-optimal_final.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
    }
    else
    {
        pParam->m_structureFile = m_pBrowseConfig->getPath().toStdString();
        pParam->m_modelFile = m_pBrowseWeights->getPath().toStdString();
        pParam->m_labelsFile = m_pBrowseLabels->getPath().toStdString();
    }
    emit doApplyProcess(m_pParam);
}
