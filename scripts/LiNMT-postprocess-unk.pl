#!/usr/bin/perl

# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * NiuTrans.NMT-postprocess-unk.pl     * #

use strict;


my $logo = "Postprocess unk\n";
print STDERR $logo;

my %param;
my %dict;

GetParameter(@ARGV);

LoadDict();
PostProcessUnk();




#################################################
sub PostProcessUnk
{
  print STDERR "Start replacing unks ...\n";
  open (INPUTSRC, "<", $param{"-source"}) or die "Error: can not read $param{'-source'}.\n";
  open (INPUTTRA, "<", $param{"-nmttrans"}) or die "Error: can not read $param{'-nmttrans'}.\n";
  open (OUTPUT, ">", $param{"-output"}) or die "Error: can not write $param{'-output'}.\n";
  
  my $line_num = 0;
  my $unk_num = 0;
  my $replace_num = 0;
  my $replace_percentage = 0;
  while (my $line_tra = <INPUTTRA>)
  {
    my $line_src = <INPUTSRC>;
    ++$line_num;
    if ($line_num % 100 == 0)
    {
      if ($unk_num ne 0)
      {
        $replace_percentage = $replace_num / $unk_num;
        $replace_percentage = sprintf "%.2f", $replace_percentage;
      }
      print STDERR "\r  $line_num, $unk_num unks, $replace_num are replaced, $replace_percentage \%";
    }

    
    $line_tra =~ s/[\r\n]//g;
    my @fields = split / \|\|\|\| /, $line_tra;

    
    if (scalar(@fields) ne 4)
    {
      print OUTPUT $line_tra."\n";
      next;
    }
    
    #print OUTPUT $line_tra."\n";
    $line_src =~ s/[\r\n]//g;
    my @src_words = split /\s+/, $line_src;
    my @tgt_words = split /\s+/, $fields[0];
    my @tgt_align = split /\s+/, $fields[2];
    if (scalar(@tgt_words) ne scalar(@tgt_align))
    {
      print STDERR "\nError: format error in $line_num\n";
    }
    
    my $replaced_unk_trans = "";
    for my $i (0..scalar(@tgt_words) - 1)
    {
      if ("<UNK>" eq $tgt_words[$i])
      {
        ++$unk_num;
        if(exists $dict{$src_words[$tgt_align[$i]]})
        {
          ++$replace_num;
          $replaced_unk_trans .= " ".$dict{$src_words[$tgt_align[$i]]};
        } else {
          if ($param{"-outoov"} eq 1)
          {
            $replaced_unk_trans .= " <$src_words[$tgt_align[$i]]>";
          }
        }
      }
      else
      {
        $replaced_unk_trans .= " ".$tgt_words[$i];
      }
    }
    $replaced_unk_trans =~ s/\s+$//g;
    $replaced_unk_trans =~ s/^\s+//g;
    print OUTPUT $replaced_unk_trans." |||| ".$fields[1]." |||| ".$fields[2]." |||| ".$fields[3]."\n";
    
  }

  if ($unk_num ne 0)
  {
    $replace_percentage = $replace_num / $unk_num;
    $replace_percentage = sprintf "%.2f", $replace_percentage;
  }
  print STDERR "\r  $line_num, $unk_num unks, $replace_num are replaced, $replace_percentage \%\n";

  close (INPUTTRA);
  close (INPUTSRC);
  close (OUTPUT);
}


#################################################
sub LoadDict()
{
  print STDERR "Start loading dict ...\n";
  open (INPUTDICT, "<", $param{"-dict"}) or die "Error: can not read $param{'-dict'}.\n";
  my $line_num = 0;
  while (my $line = <INPUTDICT>)
  {
    ++$line_num;
    $line =~ s/[\r\n]//g;
    my @fields = split / \|\|\| /, $line;
    if (scalar(@fields) ne 3)
    {
      print STDERR "\nError: format error in $line_num\n";
      next;
    }
    
    $dict{$fields[0]} = $fields[1];
    
    if ($line_num % 100000 == 0)
    {
      print STDERR "\r  $line_num";
    }
  }
  print STDERR "\r  $line_num\n";
  my $dict_size = keys %dict;
  print STDERR "  dict_size: ".$dict_size."\n";
  close (INPUTDICT);
}


#################################################
sub GetParameter
{
  if( ( scalar( @_ ) < 2 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "        NiuTrans.NMT-postprocess-unk.pl   [OPTIONS]\n".
                 "[OPTION]\n".
                 "            -dict  : top1 lexical table\n".
                 "          -source  : Source file\n".
                 "        -nmttrans  : NMT translations\n".
                 "          -output  : Translations w/o unks\n".
                 "          -outoov  : whether output oovs.\n".
                 "                       Default is 1\n".
                 "[EXAMPLE]\n".
                 "        perl NiuTrans.NMT-postprocess-unk.pl\n";
    exit( 0 );
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }

  if (!exists $param{"-dict"} || !exists $param{"-source"} || !exists $param{"-nmttrans"} || !exists $param{"-output"})
  {
    print STDERR "ERROR: '-dict', '-source', '-nmttrans', or '-output' must be assigned!\n";
    exit(1);
  }
  
  if (!exists $param{"-outoov"})
  {
    $param{"-outoov"} = 1;
  }
}


#################################################
sub Ssystem
{
  print STDERR "Running: @_\n";
  system( @_ );
  if( $? == -1 )
  {
    print STDERR "Error: Failed to execute: @_\n  $!\n";
    exit( 1 );
  }
  elsif( $? & 127 )
  {
    printf STDERR "Error: Execution of: @_\n   die with signal %d, %s coredump\n",
                  ($? & 127 ), ( $? & 128 ) ? 'with' : 'without';
    exit( 1 );
  }
  else
  {
    my $exitcode = $? >> 8;
    print STDERR "Exit code: $exitcode\n" if $exitcode;
    return ! $exitcode;
  }         
}

