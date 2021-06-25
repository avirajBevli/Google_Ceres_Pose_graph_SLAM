file(REMOVE_RECURSE
  "hw.pdb"
  "hw"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/hw.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
