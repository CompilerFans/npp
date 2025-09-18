# NPP源文件查找功能

# 递归查找C++源文件
function(npp_find_cpp_sources output_var search_dir)
    file(GLOB_RECURSE ${output_var} 
        "${search_dir}/*.cpp"
    )
    set(${output_var} ${${output_var}} PARENT_SCOPE)
endfunction()

# 递归查找CUDA源文件  
function(npp_find_cuda_sources output_var search_dir)
    file(GLOB_RECURSE ${output_var}
        "${search_dir}/*.cu"
    )
    set(${output_var} ${${output_var}} PARENT_SCOPE)
endfunction()

# 查找所有NPP源文件
function(npp_find_all_sources cpp_var cuda_var search_dir)
    npp_find_cpp_sources(${cpp_var} ${search_dir})
    npp_find_cuda_sources(${cuda_var} ${search_dir})
    set(${cpp_var} ${${cpp_var}} PARENT_SCOPE)
    set(${cuda_var} ${${cuda_var}} PARENT_SCOPE)
endfunction()

# 查找测试源文件 - 递归后缀匹配
function(npp_find_test_sources output_var test_dir)
    file(GLOB_RECURSE ${output_var}
        "${test_dir}/**.cpp"
    )
    # 只包含以test_开头的文件
    set(filtered_sources)
    foreach(source_file ${${output_var}})
        get_filename_component(filename ${source_file} NAME)
        if(filename MATCHES "^test_.*\\.cpp$")
            list(APPEND filtered_sources ${source_file})
        endif()
    endforeach()
    set(${output_var} ${filtered_sources} PARENT_SCOPE)
endfunction()