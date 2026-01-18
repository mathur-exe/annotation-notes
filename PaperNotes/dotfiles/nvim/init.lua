-- Basic quality-of-life defaults.
-- Mouse: allows click/drag to enter Visual mode and select text.
vim.opt.mouse = "a"
vim.opt.mousemodel = "extend"

-- On some terminals/keyboards (notably macOS), the key labeled "Delete" may be sent as <BS>.
-- Make both <Del> and <BS> delete the current Visual selection.
vim.keymap.set("v", "<Del>", "d", { silent = true })
vim.keymap.set("v", "<BS>", "d", { silent = true })

