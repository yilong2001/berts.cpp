include_directories(${CMAKE_SOURCE_DIR}/)

if (MSVC)
    if (BERT_AVX2)
        add_compile_options(/arch:AVX2)
    elseif (BERT_AVX)
        add_compile_options(/arch:AVX)
    endif()
endif()

if(WIN32)
add_definitions(-DNOMINMAX=1)
endif()


add_executable(bert-main bert-main.cpp)
if(NOT WIN32)
    target_link_libraries(bert-main PRIVATE bertencoder bertbase tokenization ggml utf8proc)
else()
    target_link_libraries(bert-main PRIVATE bertencoder bertbase tokenization ggml utf8proc ws2_32)
endif()

add_executable(bert-rest bert-rest.cpp)
if(NOT WIN32)
	target_link_libraries(bert-rest PRIVATE bertencoder bertbase tokenization ggml utf8proc oatpp)
else()
	target_link_libraries(bert-rest PRIVATE bertencoder bertbase tokenization ggml utf8proc oatpp ws2_32)
endif()
