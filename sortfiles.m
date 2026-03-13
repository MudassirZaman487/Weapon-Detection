
function sorted_files = sortfiles(files)
    % Sort files by name for consistency
    [~, idx] = sort({files.name});
    sorted_files = files(idx);
end
