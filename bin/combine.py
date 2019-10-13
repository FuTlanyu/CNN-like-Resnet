import multiresolutionimageinterface as mir


for i in range(4, 9):
    print(i)

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open('F:/LungCancerData/testForHeatmap/test/test_wsi/' + str(i) + '.tif')
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource('F:/LungCancerData/training1/Annotations/' + str(i) + '.xml')
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    output_path = 'F:/LungCancerData/testForHeatmap/test/test_mask/' + str(i) + '.tif'
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing())