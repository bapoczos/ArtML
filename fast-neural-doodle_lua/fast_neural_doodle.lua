require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'hdf5'
require 'loadcaffe'
require 'src/style_loss'
require 'src/content_loss'

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-masks_hdf5', 'masks.hdf5', 
           'Path to .hdf5 file with masks. It can be obtained with get_mask_hdf5.py.')

-- Optimization options
cmd:option('-content_weight', 1e-3)
cmd:option('-style_weight', 1e0)
cmd:option('-tv_weight', 0, 'TV weight, zero works fine for me.')
cmd:option('-normalize_gradients', false)
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

cmd:option('-num_iterations', '1000',
           'Comma separated (no spaces) list with iteration number to do at corresponding resolution.')
cmd:option('-resolutions', '0', 'Comma separated (no spaces) list or resolutions. 0 for original')

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 50)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'avg', 'max|avg')
cmd:option('-proto_file', 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-vgg_no_pad', false, 'Because of border effects padding is advised to be set to `valid`. This flag does it.')
cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

local function main()
  init = true
   
  -- Load images
  local f_data = hdf5.open(params.masks_hdf5)
  local style_img = f_data:read('style_img'):all():float()
  if cur_resolution ~= 0 then 
    style_img = image.scale(style_img, cur_resolution, cur_resolution)
  end
  style_img = preprocess(style_img):float()

  local has_content = f_data:read('has_content'):all()[1] == 1
  local content_img = nil
  if has_content then
    content_img = f_data:read('content_img'):all():float()
    if cur_resolution ~= 0 then
        content_img =  image.scale(content_img, cur_resolution, cur_resolution)
    end
    content_img = preprocess(content_img):float()
  else
    print('Content image is not provided, content loss will be ignored')
    params.content_weight = 0
  end

  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      style_img = style_img:cuda()
      if has_content then
        content_img = content_img:cuda()
      end
    else
      style_img = style_img:cl()
      if has_content then
        content_img = content_img:cl()
      end
    end
  end

  local n_colors = f_data:read('n_colors'):all()[1]

  -- Load masks
  local style_masks, target_masks = {}, {}
  for k = 0, n_colors - 1 do
    local style_mask = f_data:read('style_mask_' .. k):all():float()
    local target_mask = f_data:read('target_mask_' .. k):all():float()
    
    -- Scale
    if cur_resolution ~= 0 then 
      style_mask =  image.scale(style_mask,  cur_resolution, cur_resolution, 'simple')
      target_mask = image.scale(target_mask, cur_resolution, cur_resolution, 'simple')
    end
    table.insert(style_masks, style_mask)
    table.insert(target_masks, target_mask)
  end
  local target_size = target_masks[1]:size()
 
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()

  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
        tv_mod:cuda()
      else
        tv_mod:cl()
      end
    end
    net:add(tv_mod)
  end
  for i = 1, #cnn do
    
    if next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      local is_conv =  (layer_type == 'nn.SpatialConvolution' or layer_type == 'cudnn.SpatialConvolution')
     
      if is_pooling then
       
        if params.pooling == 'avg' then
          assert(layer.padW == 0 and layer.padH == 0)
          local kW, kH = layer.kW, layer.kH
          local dW, dH = layer.dW, layer.dH
          local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
          if params.gpu >= 0 then
            if params.backend ~= 'clnn' then
              avg_pool_layer:cuda()
            else
              avg_pool_layer:cl()
            end
          end
          local msg = 'Replacing max pooling at layer %d with average pooling'
          print(string.format(msg, i))
          
          layer = avg_pool_layer
        end

        layer:floor()
        for k, _ in ipairs(style_masks) do
          style_masks[k] = image.scale(style_masks[k]  , math.floor(style_masks[k]:size(2)/2), math.floor(style_masks[k]:size(1)/2))
          target_masks[k] = image.scale(target_masks[k] , math.floor(target_masks[k]:size(2)/2), math.floor(target_masks[k]:size(1)/2))
        end

        style_masks = deepcopy(style_masks)
        target_masks = deepcopy(target_masks)

      elseif is_conv then

        local sap = nn.SpatialAveragePooling(3,3,1,1,1,1):float()
        for k, _ in ipairs (style_masks) do
            style_masks[k] = sap:forward(style_masks[k]:add_dummy())[1]:clone()
            target_masks[k] = sap:forward(target_masks[k]:add_dummy())[1]:clone()
        end
        
        -- Turn off padding
        if params.vgg_no_pad and (layer_type == 'nn.SpatialConvolution' or layer_type == 'cudnn.SpatialConvolution') then
          layer.padW = 0
          layer.padH = 0

          for k, _ in ipairs (style_masks) do
            style_masks[k] = image.crop(style_masks[k] , 'c', style_masks[k]:size(2)-2, style_masks[k]:size(1)-2) 
            target_masks[k] = image.crop(target_masks[k] , 'c', target_masks[k]:size(2)-2, target_masks[k]:size(1)-2) 
          end
          style_masks = deepcopy(style_masks)
          target_masks = deepcopy(target_masks)
        end
      end

      net:add(layer)
      
      -- Content
      if has_content and name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local target = net:forward(content_img):clone()
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      -- Style   
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            gram = gram:cuda()
          else
            gram = gram:cl()
          end
        end

        local target_features = net:forward(style_img):clone()
              
        -- Compute target gram mats
        local target_grams = {}
        for k, _ in ipairs(style_masks) do
          local layer_mask = style_masks[k]:add_dummy():expandAs(target_features)
          if params.gpu >= 0 then
            if params.backend ~= 'clnn' then
              layer_mask = layer_mask:cuda()
            else
              layer_mask = layer_mask:cl()
            end
          end
          local masked = torch.cmul(target_features, layer_mask)
       
          local target = gram:forward(masked):clone()

          if style_masks[k]:mean() > 0 then
            target:div(target_features:nElement() * style_masks[k]:mean())
          end

          target_grams[k] = target
        end

        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target_grams, norm,  deepcopy(target_masks)):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end

        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end
  init = false
  -- We don't need the base CNN anymore, so clean it up to save memory.
  
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
  
  -- Initialize with previous or with noise
  if img then 
    img = image.scale(img:float(), target_size[2], target_size[1])
  else
    if params.seed >= 0 then
      torch.manualSeed(params.seed)
    end
    img = torch.randn(3, target_size[1], target_size[2]):float():mul(0.001)
  end

  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      img = img:cuda()
    else
      img = img:cl()
    end
  end
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = cur_num_iterations,
            tolX = -1,
      tolFun = -1,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this fucntion many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:updateGradInput(x, dy)
    local loss = 0
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss, style_losses)
    maybe_save(num_calls, img)

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, cur_num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end
end
  


-------------------------------------------------------------

params = cmd:parse(arg)

-- Load libs
if params.gpu >= 0 then
  if params.backend ~= 'clnn' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu + 1)
  else
    require 'clnn'
    require 'cltorch'
    cltorch.setDevice(params.gpu + 1)
  end
else
  params.backend = 'nn'
end
require 'src/utils'

if params.backend == 'cudnn' then
  require 'cudnn'
  if params.cudnn_autotune then
    cudnn.benchmark = true
  end
end

-- Load VGG
local loadcaffe_backend = params.backend
if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
if params.gpu >= 0 then
  if params.backend ~= 'clnn' then
    cnn:cuda()
  else
    cnn:cl()
  end
end

for i = 1,9 do
  cnn:remove()
end


-- run at different resolutions
local resolutions = params.resolutions:split(",")
local num_iterations = params.num_iterations:split(",")
assert(#resolutions == #num_iterations, 'Incorrect resolution-iteration correspondence.')

img = nil
for res = 1, #resolutions do
  cur_resolution = tonumber(resolutions[res])
  cur_num_iterations = tonumber(num_iterations[res])

  main(params)
end
